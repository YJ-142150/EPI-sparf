import torch
from easydict import EasyDict as edict
from typing import Callable, Sequence, List, Mapping, MutableMapping, Tuple, Union, Dict
from typing import Any, Optional
from source.utils.camera import pose_inverse_4x4
from source.utils.config_utils import override_options
from source.training.core.base_epipolar_loss import EpipolarBasedLoss
from source.utils.camera import pose_inverse_4x4
from source.utils.geometry.batched_geometry_utils import batch_project_to_other_img


class CorrespondencesPairRenderDepthAndGet3DPtsAndReprojectAndEpipolar(EpipolarBasedLoss):
    """The main class for the correspondence loss of SPARF. It computes the re-projection error
    between previously extracted correspondences relating the input views. The projection
    is computed with the rendered depth from the NeRF and the current camera pose estimates.
    """

    def __init__(self, opt: Dict[str, Any], nerf_net: torch.nn.Module, flow_net: torch.nn.Module,
                 train_data: Dict[str, Any], device: torch.device):
        super().__init__(opt, nerf_net, flow_net, train_data, device)
        default_cfg = edict({'diff_loss_type': 'huber',
                             'compute_photo_on_matches': False,
                             'renderrepro_do_pixel_reprojection_check': False,
                             'renderrepro_do_depth_reprojection_check': False,
                             'renderrepro_pixel_reprojection_thresh': 10.,
                             'renderrepro_depth_reprojection_thresh': 0.1,
                             'use_gt_depth': False,  # debugging
                             'use_gt_correspondences': False,  # debugging
                             'use_dummy_all_one_confidence': False  # debugging
                             })
        self.opt = override_options(self.opt, default_cfg)
        self.opt = override_options(self.opt, opt)

    def compute_render_and_repro_loss_w_repro_thres(self, opt: Dict[str, Any], pixels_in_self_int: torch.Tensor,
                                                    depth_rendered_self: torch.Tensor, intr_self: torch.Tensor,
                                                    pixels_in_other: torch.Tensor, depth_rendered_other: torch.Tensor,
                                                    intr_other: torch.Tensor, T_self2other: torch.Tensor,
                                                    conf_values: torch.Tensor, stats_dict: Dict[str, Any],
                                                    return_valid_mask: bool = False
                                                    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Computes the actual re-projection error loss between 'self' and 'other' images,
        along with possible filterings.

        Args:
            opt (edict): settings
            pixels_in_self_int (torch.Tensor): (N, 2)
            depth_rendered_self (torch.Tensor): (N)
            intr_self (torch.Tensor): (3, 3)
            pixels_in_other (torch.Tensor): (N, 2)
            depth_rendered_other (torch.Tensor): (N)
            intr_other (torch.Tensor): (3, 3)
            T_self2other (torch.Tensor): (4, 4)
            conf_values (torch.Tensor): (N, 1)
            stats_dict (dict): dict to keep track of statistics to be logged
            return_valid_mask (bool, optional): Defaults to False.
        """
        pts_self_repr_in_other, depth_self_repr_in_other = batch_project_to_other_img(
            pixels_in_self_int.float(), di=depth_rendered_self,
            Ki=intr_self, Kj=intr_other, T_itoj=T_self2other, return_depth=True)

        loss = torch.norm(pts_self_repr_in_other - pixels_in_other, dim=-1, keepdim=True)  # [N_rays, 1]
        valid = torch.ones_like(loss).bool()
        if opt.renderrepro_do_pixel_reprojection_check:
            valid_pixel = loss.detach().le(opt.renderrepro_pixel_reprojection_thresh)
            valid = valid & valid_pixel
            stats_dict['perc_val_pix_rep'] = valid_pixel.sum().float() / (valid_pixel.nelement() + 1e-6)

        if opt.renderrepro_do_depth_reprojection_check:
            valid_depth = torch.abs(depth_rendered_other - depth_self_repr_in_other) / (depth_rendered_other + 1e-6)
            valid_depth = valid_depth.detach().le(opt.renderrepro_depth_reprojection_thresh)
            valid = valid & valid_depth.unsqueeze(-1)
            stats_dict['perc_val_depth_rep'] = valid_depth.sum().float() / (valid_depth.nelement() + 1e-6)

        loss_corres = self.compute_diff_loss(loss_type=opt.diff_loss_type,
                                             diff=pts_self_repr_in_other - pixels_in_other,
                                             weights=conf_values, mask=valid, dim=-1)

        if return_valid_mask:
            return loss_corres, stats_dict, valid
        return loss_corres, stats_dict

    def compute_loss_on_image_pair(self, data_dict: Dict[str, Any], images: torch.Tensor, poses_w2c: torch.Tensor,
                                   intrs: torch.Tensor, id_self: int, id_matching_view: int,
                                   corres_map_self_to_other: torch.Tensor,
                                   corres_map_self_to_other_rounded_flat: torch.Tensor,
                                   conf_map_self_to_other: torch.Tensor, mask_correct_corr: torch.Tensor,
                                   pose_w2c_self: torch.Tensor, pose_w2c_other: torch.Tensor, intr_self: torch.Tensor,
                                   intr_other: torch.Tensor,
                                   loss_dict: Dict[str, Any], stats_dict: Dict[str, Any], plotting_dict: Dict[str, Any]
                                   ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Core of the loss function.
        Args:
            data_dict (edict): Input data dict. Contains important fields:
                                - Image: GT images, (B, 3, H, W)
                                - intr: intrinsics (B, 3, 3)
                                - pose: GT w2c pose (B, 3, 4)
                                - poses_w2c: current estimate of w2c poses (being optimized)
                                - idx: idx of the images (B)
                                - depth_gt (optional): gt depth, (B, 1, H, W)
                                - valid_depth_gt (optional): (B, 1, H, W)
            images: Input images (B, H, W, 3)
            poses_w2c: (B, 3, 4), current estimates of the poses
            intrs: image intrinsics (B, 3, 3)
            id_self
            id_matching_view
            corres_map_self_to_other (H, W, 2) float Tensor
            corres_map_self_to_other_rounded_flat: Contain idx of match in flattened (HW) tensor.
                                                   Shape is (H, W, 1) Long Tensor
            conf_map_self_to_other (H, W)
            mask_correct_corr: (H, W, 1), bool Tensor
            pose_w2c_self: (4, 4)
            pose_w2c_other (4, 4)
            intr_self: (3, 3)
            intr_other: (3, 3)
            loss_dict: dictionary containing the losses
            stats_dict: dictionary containing the stats
            plotting_dict: dictionary containing plots
            plot: bool
            compute_sampling_loss: bool
        """
        # the actual render and reproject code
        iteration = data_dict['iter']
        B, H, W = images.shape[:3]

        pixels_in_self = self.grid[mask_correct_corr]  # [N_ray, 2], absolute pixel locations
        ray_in_self_int = self.grid_flat[mask_correct_corr]  # [N_ray]

        pixels_in_other = corres_map_self_to_other[mask_correct_corr]  # [N_ray, 2], absolute pixel locations
        ray_in_other_int = corres_map_self_to_other_rounded_flat[mask_correct_corr]  # [N_ray]
        conf_values = conf_map_self_to_other[mask_correct_corr]  # [N_ray, 1]

        # in case there are too many values, subsamples
        if ray_in_self_int.shape[0] > self.opt.nerf.rand_rays // 2:
            random_values = torch.randperm(ray_in_self_int.shape[0], device=self.device)[:self.opt.nerf.rand_rays // 2]
            ray_in_self_int = ray_in_self_int[random_values]
            pixels_in_self = pixels_in_self[random_values]

            pixels_in_other = pixels_in_other[random_values]
            ray_in_other_int = ray_in_other_int[random_values]
            conf_values = conf_values[random_values]

        ret_self = self.net.render_image_at_specific_pose_and_rays(self.opt, data_dict, pose_w2c_self[:3],
                                                                   intr_self, H, W,
                                                                   pixels=pixels_in_self, mode='train', iter=iteration)
        # edict with rgb_fine=rgb_fine,depth_fine=depth_fine,opacity_fine=opacity_fine, shape [1, N_rays, K]

        ret_other = self.net.render_image_at_specific_pose_and_rays(self.opt, data_dict, pose_w2c_other[:3],
                                                                    intr_other, H, W,
                                                                    pixels=pixels_in_other, mode='train',
                                                                    iter=iteration)

        if self.opt.compute_photo_on_matches:
            # compute the rgb loss for the two images
            image_self = images[id_self].view(-1, 3)[ray_in_self_int]
            image_other = images[id_matching_view].view(-1, 3)[ray_in_other_int]
            loss_photo_self = self.MSE_loss(ret_self.rgb.view(-1, 3), image_self)
            loss_photo_other = self.MSE_loss(ret_other.rgb.view(-1, 3), image_other)

            if 'rgb_fine' in ret_self.keys():
                loss_photo_self += self.MSE_loss(ret_self.rgb_fine.view(-1, 3), image_self)
                loss_photo_other += self.MSE_loss(ret_other.rgb_fine.view(-1, 3), image_other)
            loss_photo = (loss_photo_other + loss_photo_self) / 2.
            loss_dict['render_matches'] += loss_photo

        # compute the correspondence loss
        # for each image, project the pixel to the other image according to current estimate of pose
        depth_rendered_self = ret_self.depth.squeeze(0).squeeze(-1)
        depth_rendered_other = ret_other.depth.squeeze(0).squeeze(-1)
        # ret_self.depth is [1, N_ray, 1], it needs to be [N_ray]
        stats_dict['depth_in_corr_loss'] = depth_rendered_self.detach().mean()

        T_self2other = pose_w2c_other @ pose_inverse_4x4(pose_w2c_self)

        # [N_ray, 2] and [N_ray]
        loss_corres, stats_dict = self.compute_render_and_repro_loss_w_repro_thres \
            (self.opt, pixels_in_self, depth_rendered_self, intr_self,
             pixels_in_other, depth_rendered_other, intr_other, T_self2other,
             conf_values, stats_dict)

        loss_corres_, stats_dict = self.compute_render_and_repro_loss_w_repro_thres \
            (self.opt, pixels_in_other, depth_rendered_other, intr_other,
             pixels_in_self, depth_rendered_self, intr_self, pose_inverse_4x4(T_self2other),
             conf_values, stats_dict)
        loss_corres += loss_corres_
        
        # compute the epipolar loss
        epi_constraint=True
        if epi_constraint:
            #E = self.compute_essential_matrix(self.opt, pose_w2c_self, pose_w2c_other)

            #mean_x_self = pixels_in_self[:, 0].mean()
            #mean_y_self = pixels_in_self[:, 1].mean()
            #std_x_self = pixels_in_self[:, 0].std()
            #std_y_self = pixels_in_self[:, 1].std()

            #mean_x_other = pixels_in_other[:, 0].mean()
            #mean_y_other = pixels_in_other[:, 1].mean()
            #std_x_other = pixels_in_other[:, 0].std()
            #std_y_other = pixels_in_other[:, 1].std()

            #pixels_in_self[:, 0] = (pixels_in_self[:, 0]) / H #- mean_x_self) / std_x_self
            #pixels_in_self[:, 1] = (pixels_in_self[:, 1]) / W # - mean_y_self) / std_y_self

            #pixels_in_other[:, 0] = (pixels_in_other[:, 0]) / H # - mean_x_other) / std_x_other
            #pixels_in_other[:, 1] = (pixels_in_other[:, 1]) / W # - mean_y_other) / std_y_other

            #intr_self_inv = torch.inverse(intr_self)
            bearing_pixels_in_self = self.compute_bearing_vectors(self.opt, pixels_in_self, intr_self, pose_w2c_self[:3, :3]) # intr_self_inv)

            #intr_other_inv = torch.inverse(intr_other)
            bearing_pixels_in_other = self.compute_bearing_vectors(self.opt, pixels_in_other,intr_other, pose_w2c_other[:3, :3]) #intr_other_inv)

            residuals = self.energy_function_without_uncertainty(self.opt, pose_w2c_self, pose_w2c_other, bearing_pixels_in_self, bearing_pixels_in_other, conf_values)
            loss_epi = residuals # * conf_values[:, 0] # -loss_epi_real
            #print(conf_values[:,0])
            Energy = torch.sum(loss_epi)

            loss_dict['epipolar'] += Energy


        if 'depth_fine' in ret_other.keys():
            depth_rendered_self = ret_self.depth_fine.squeeze(0).squeeze(-1)
            depth_rendered_other = ret_other.depth_fine.squeeze(0).squeeze(-1)

            loss_corres_, stats_dict = self.compute_render_and_repro_loss_w_repro_thres \
                (self.opt, pixels_in_self, depth_rendered_self, intr_self,
                 pixels_in_other, depth_rendered_other, intr_other, T_self2other,
                 conf_values, stats_dict)
            loss_corres += loss_corres_

            loss_corres_, stats_dict = self.compute_render_and_repro_loss_w_repro_thres \
                (self.opt, pixels_in_other, depth_rendered_other, intr_other,
                 pixels_in_self, depth_rendered_self, intr_self, pose_inverse_4x4(T_self2other),
                 conf_values, stats_dict)
            loss_corres += loss_corres_
        loss_corres = loss_corres / 4. if 'depth_fine' in ret_other.keys() else loss_corres / 2.
        loss_dict['corres'] = loss_corres
        return loss_dict, stats_dict, plotting_dict

    def compute_essential_matrix(self, opt, pose_w2c_self, pose_w2c_other):
        """
        Compute the essential matrix between two poses.
        Args:
            pose_w2c_self: (4, 4) tensor
            pose_w2c_other: (4, 4) tensor
        Returns:
            E: (3, 3) tensor
        """
        # Extract rotation (R) and translation (t) from the poses
        R_self = pose_w2c_self[:3, :3]
        t_self = pose_w2c_self[:3, 3]
        R_other = pose_w2c_other[:3, :3]
        t_other = pose_w2c_other[:3, 3]

        # Compute relative rotation and translation between the two cameras
        R = torch.matmul(R_other, R_self.T)
        t = t_other - torch.matmul(R, t_self)

        # Construct the essential matrix from R and t
        #t_x = torch.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
        t = torch.tensor([t[0], t[1], t[2]], device='cuda:0')  # Assuming t is a list or array of length 3
        t_x = torch.zeros((3, 3))
        t_x[0, 1], t_x[0, 2], t_x[1, 0], t_x[1, 2], t_x[2, 0], t_x[2, 1] = -t[2], t[1], t[2], -t[0], -t[1], t[0]
        t_x = t_x.to('cuda')
        E = torch.matmul(t_x, R)

        return E

    def compute_bearing_vectors(self, opt, pixel_coords, K, R):  #K_inv):
        """
        Compute the bearing vectors from pixel coordinates.

        Args:
        - pixel_coords (torch.Tensor): the [N, 2] tensor containing N pixel coordinates.
        - K_inv (torch.Tensor): the [3, 3] intrinsic inverse matrix.

        Returns:
        - bearing_vectors (torch.Tensor): the [N, 3] tensor containing N bearing vectors.
        """
        # Homogenize pixel coordinates
        #ones = torch.ones(pixel_coords.shape[0], 1, device='cuda:0')
        #homogenized_pixels = torch.cat([pixel_coords, ones], dim=1)  # [N, 3]

        # Convert pixel coordinates to camera coordinates
        #cam_coords = torch.matmul(homogenized_pixels, K_inv.T)  # [N, 3]

        # Normalize to get the bearing vectors
        #norms = torch.norm(cam_coords, dim=1, keepdim=True)
        #bearing_vectors = cam_coords / norms
        # Convert to tensor if not already

        #image_points = torch.tensor(pixel_coords, dtype=torch.float32)

        # Append ones for homogeneous coordinates
        #image_points_homogeneous = torch.cat([pixel_coords, torch.ones(pixel_coords.shape[0], 1, device='cuda:0')], dim=1)
        points = torch.cat([pixel_coords, torch.ones(pixel_coords.shape[0], 1, device='cuda:0')], dim=1)
        # 1. Convert to normalized image plane
        K_inv = torch.inverse(torch.tensor(K, dtype=torch.float32))
        #normalized_image_points = torch.matmul(K_inv, image_points_homogeneous.t()).t()[:,:2]

        # 2. Convert to homogeneous coordinates (in camera frame)
        #camera_coords = torch.cat([normalized_image_points, torch.ones(normalized_image_points.shape[0], 1,device='cuda:0')], dim=1)

        # 3. Convert to world coordinates using the rotation matrix R
        #world_coords = torch.matmul(torch.tensor(R, dtype=torch.float32), camera_coords.t()).t()

        # 4. Normalize to get unit feature vectors in world coordinates
        #unit_feature_vectors_world = world_coords / world_coords.norm(dim=1, keepdim=True)

        transformed_points = torch.einsum("...ij,...nj->...ni", K_inv, points)
        norms = torch.norm(transformed_points, dim=-1)
        transformed_points = transformed_points / norms[..., None]
        return transformed_points #unit_feature_vectors_world #bearing_vectors

    def energy_function_without_uncertainty(self, opt, pose_w2c_self, pose_w2c_other, bearing_pixels_in_self, bearing_pixels_in_other, confidences):
        # Extracting rotation and translation from the poses
        R_self = pose_w2c_self[:3, :3]
        t_self = pose_w2c_self[:3, 3]
        R_other = pose_w2c_other[:3, :3]
        t_other = pose_w2c_other[:3, 3]

        # Compute relative rotation and translation between the two cameras
        R_rel = torch.matmul(R_other, R_self.T)
        t_rel = -torch.matmul(R_rel, t_self) + t_other
        #unit_t_rel = t_rel / t_rel.norm(keepdim=True)
        
        #sigma_values = 1.0 - confidences
        cov_values = 1.0 / confidences
        cov_matrices = torch.zeros(512, 3, 3)

        for i in range(512):
            cov_matrices[i] = torch.diag(cov_values[i] * torch.ones(3, device='cuda:0'))

        # Transform all bearing_pixels in the second image
        transformed_bearing_pixels_in_other = torch.matmul(R_rel, bearing_pixels_in_other.T).T
        #print('transformed', transformed_bearing_pixels_in_other.size())
        # Compute the cross products for all vectors at once
        #cross_products = torch.cross(bearing_pixels_in_self, transformed_bearing_pixels_in_other)
        skew_bearing_pixels_in_self = self.skewmat(self.opt, bearing_pixels_in_self)
        #print('skew', skew_bearing_pixels_in_self.size())
        cross_product = torch.einsum('bij, bj -> bi', skew_bearing_pixels_in_self, transformed_bearing_pixels_in_other) #torch.matmul(skew_bearing_pixels_in_self, transformed_bearing_pixels_in_other)
        #print('t_rel', t_rel.size())
        #print('cross_product', cross_product.size())
        #print(torch.einsum('i, bi -> b', t_rel, cross_product).size())

        fin = torch.einsum('i, bi -> b', t_rel, cross_product)
        # Compute the dot products for all residuals at once
        #residuals = torch.sum(fin ** 2)
        epi_zeros = torch.zeros_like(fin)
        residuals = torch.nn.functional.huber_loss(fin, epi_zeros, reduction='none', delta=1.)
        
        #E = torch.sum(residuals)

        # Compute the skew-symmetric matrices (외적 행렬) for bearing_pixels_in_self all at once
       
        # zero = torch.zeros(bearing_pixels_in_self.size(0), device='cuda:0') #.to(bearing_pixels_in_self.device)
        '''f_hat_self = torch.stack([
            zero, -bearing_pixels_in_self[:, 2], bearing_pixels_in_self[:, 1],
            bearing_pixels_in_self[:, 2], zero, -bearing_pixels_in_self[:, 0],
            -bearing_pixels_in_self[:, 1], bearing_pixels_in_self[:, 0], zero
        ], dim=1).reshape(-1, 3, 3)'''
        '''f_hat_self = self.skewmat(self, bearing_pixels_in_self)

        # Compute denominator for each residual
        #temp = torch.matmul(torch.matmul(torch.matmul(t_rel.T.unsqueeze(0).expand_as(skew_matrices), skew_matrices), R_rel), sigmas)
        #tiemp = torch.matmul(temp, skew_matrices.transpose(1, 2))

        #denominators = torch.sum(temp * t_rel, dim=-1)
        #denominator = torch.einsum('bi, bij, bjk, bkl, bl -> b', t_rel, f_hat_self, R_rel, sigmas, R_rel.T, f_hat_self.T, t_rel)
        #print(t_rel.size())
        #print(f_hat_self.size())
        #print(R_rel.size())
        #print(sigmas.size())
        denominator = torch.einsum('i, bij, jk, bkl, lm, bno, o -> b', t_rel, f_hat_self, R_rel, cov_matrices, R_rel.T, f_hat_self.T, t_rel)


        # Avoid zero denominators
        denominators = torch.where(denominators != 0, denominators, torch.ones_like(denominators))

        print('size of resi', residuals.size())
        print('size of deno', denominators.size())
        weighted_residuals = residuals / denominators
        '''
        return residuals

    def skewmat(self, opt, x_vec):
        '''
        torch.matrix_exp(a)
        Eigen::Matrix3f mat = Eigen::Matrix3f::Zero();

        mat(0, 1) = -v[2]; mat(0, 2) = +v[1];
        mat(1, 0) = +v[2]; mat(1, 2) = -v[0];
        mat(2, 0) = -v[1]; mat(2, 1) = +v[0];

        return mat;

        input : (*, 3)
        output : (*, 3, 3)
        '''

        W_row0 = torch.tensor([0.,0.,0.,  0.,0.,1.,  0.,-1.,0.]).view(3,3).to(x_vec.device)

        W_row1  = torch.tensor([0.,0.,-1.,  0.,0.,0.,  1.,0.,0.]).view(3,3).to(x_vec.device)

        W_row2  = torch.tensor([0.,1.,0.,  -1.,0.,0.,  0.,0.,0.]).view(3,3).to(x_vec.device)

        x_skewmat = torch.stack([torch.matmul(x_vec, W_row0.t()) , torch.matmul(x_vec, W_row1.t()), torch.matmul(x_vec, W_row2.t())] , dim = -1)

        return x_skewmat
