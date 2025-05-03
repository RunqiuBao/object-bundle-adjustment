import torch
from kornia.geometry import axis_angle_to_rotation_matrix, quaternion_to_rotation_matrix, rotation_matrix_to_angle_axis, axis_angle_to_quaternion, rotation_matrix_to_quaternion
from .optimizer import lm_optimize
import copy

def projection(X, r, t):
    R = axis_angle_to_rotation_matrix(r[None])[0]
    if len(X.shape) > 1:#TODO: don't want this
        x = (R @ X.mT).mT + t[None]
    else:        
        x = (R @ X) + t
    return x[...,:2] / x[...,[2]]

def stereo_projection(X, r, t, stereo_baseline):
    x_im_l = projection(X, r, t)
    
    R = axis_angle_to_rotation_matrix(r[None])[0]
    R_rl = torch.eye(3, device = X.device)
    t_rl = torch.Tensor([-stereo_baseline, 0, 0]).to(X.device)
    R_rw = R_rl @ R
    t_rw = R_rl @ t + t_rl
    if len(X.shape) > 1:
        x_im_r = (R_rw @ X.mT).mT + t_rw[None]
    else:
        x_im_r = (R_rw @ X) + t_rw
    x_im_r = x_im_r[..., :2] / x_im_r[..., [2]]
    return torch.cat((x_im_l, x_im_r[..., 0].unsqueeze(-1)), dim = -1)

def calibrated_residuals(X, theta, x_im):
    r, t = theta.chunk(2)
    x_im_hat = projection(X, r, t)
    r_im = x_im_hat - x_im
    return r_im

def calibrated_stereo_residuals(X, theta, x_im_stereo, stereo_baseline):
    """
    Args:
        X: (N, 3)
        theta: (6,)
        x_im: (N, 3)
        stereo_baseline:
    """
    r, t = theta.chunk(2)
    x_im_stereo_hat = stereo_projection(X, r, t, stereo_baseline)
    r_im_stereo = x_im_stereo_hat - x_im_stereo
    return r_im_stereo

def simple_pinhole_residuals(X, theta, x_im):
    intrinsics, r, t = theta.chunk(3)
    principal_point = intrinsics[1:]
    f = intrinsics[0]
    r_im = f*projection(X, r, t) + principal_point - x_im
    return r_im

# colmap: https://github.com/colmap/colmap/blob/f5597bf7abc2bdfa3f99b8cbeed89173062bdbfe/src/colmap/sensor/models.h#L775
def simple_radial_residuals(X, theta, x_im):
    intrinsics, r, t = theta[:4], theta[4:7], theta[7:]
    principal_point = intrinsics[1:3]
    f = intrinsics[0]
    k = intrinsics[3]
    x = projection(X, r, t)
    radius_squared = x.pow(2).sum(dim=-1, keepdim = True)
    x = x * (1 + k * radius_squared)
    r_im = f*x + principal_point - x_im
    return r_im

def optimize_calibrated(X_0, r_0, t_0, observations, dtype=torch.float32, L_0 = 1e-2, num_steps = 5):
    theta_0 = torch.cat((r_0, t_0), dim = -1)
    X_hat, theta_hat = lm_optimize(calibrated_residuals, X_0, theta_0, observations, dtype=dtype, L_0 = L_0, num_steps = num_steps)
    return X_hat, theta_hat

def optimize_simple_radial(X_0, f, principal_point, k, r_0, t_0, observations, dtype=torch.float32, L_0 = 1e-2, num_steps = 5):
    """
    Args:
        X_0: initial guess of 3D points, shape (N, 3)
        f: initial guess of focal length, shape (2, 1)
        principal_point: initial guess of principal point, shape (2, 1)
        k: parameter used to produce radial residual, see simple_radial_residuals. shape (2, 1)
        r_0: initial guess of rotation, shape (2, 3)
        t_0: initial guess of translation, shape (2, 3)
        observations: list of tuples of (p_im, indices). p_im are the projections of the 3D points in the image, shape (M, 2).
    """
    theta_0 = torch.cat((f, principal_point, k, r_0, t_0), dim = -1)
    X_hat, theta_hat = lm_optimize(simple_radial_residuals, X_0, theta_0, observations, dtype=dtype, L_0 = L_0, num_steps = num_steps)
    return X_hat, theta_hat

def optimize_obj_lm(
    poses,
    obj_2dkeypts,  # (NumObjects, NumKeypts, 3)
    uniqueobj_3dkeypts_inworld,  # (NumUniqueObjects, 4, 3)
    uniqueobj_3dcorners_inworld,  # (NumUniqueObjects, 16, 3)
    uniqueobj_labels,
    obj_uniqueobj_indices,  # (NumObj,)
    objlabel_numkeypts,
    objlabel_numcorners,
    obj_labels,
    obj_ii,
    obj_jj,
    obj_kk,
    targetObjectIndices,
    intrinsics,
    t0,
    t1,
    dtype=torch.float32,
    L_0=1e-2,
    num_steps=5
):
    """
    Args:
        poses: (:, 7).
        obj_3dkeypts: (NumObjects, 4, 3), keypts in each of 3d bounding volume. 4 means maximum 4 keypts allowed for an object.
        objlabel_numkeypts: (NumLabels,), number of keypoints in each object instance.
        obj_labels: (NumObjects,), labels of all the object instances.
        obj_ii: (NumEdges,).
        obj_jj: (NumEdges,).
        obj_kk: (NumEdges,).
        targetObjectIndices: (NumEdges,).
        intrinsics: (1, 5). The params are (fx, fy, cx, cy, baseline).
        t0: index of start frame in poses for optimization.
        t1: index of end frame in poses for optimization. I.e. optimizing poses from t0 to (t1 - 1).
        dtype:
        L_0: lambda for Levenberg-marquart.
        num_steps: number of iteration in Levenberg-marquart.
    """
    pts_vector = []
    pts_uniqueobjindices_vector = []
    pts_corners_uniqueobjindices_vector = []
    pts_keyptindices_mask = []  # last keypoint of each object is the one to be optimized.
    device = uniqueobj_3dkeypts_inworld.device
    uniqueobj_sumkeypts = []
    sumkeypts = 0
    for indexUniqueObj in range(uniqueobj_3dkeypts_inworld.shape[0]):
        numkeypts = objlabel_numkeypts[uniqueobj_labels[indexUniqueObj]].item()
        numCorners = objlabel_numcorners[uniqueobj_labels[indexUniqueObj]].item()
        oneobj_uniquekeypts = uniqueobj_3dkeypts_inworld[indexUniqueObj][:numkeypts].clone()
        if oneobj_uniquekeypts.ndim == 1:
            oneobj_uniquekeypts = oneobj_uniquekeypts[None, :]
        pts_vector.append(oneobj_uniquekeypts)
        pts_uniqueobjindices_vector.append(
            torch.Tensor([indexUniqueObj] * numkeypts).to(device=device, dtype=torch.long)
        )
        pts_corners_uniqueobjindices_vector.append(
            torch.Tensor([indexUniqueObj] * numCorners).to(device=device, dtype=torch.long)
        )
        one_mask = torch.zeros(numkeypts, device=device, dtype=torch.bool)
        one_mask[-1] = 1
        pts_keyptindices_mask.append(one_mask)
        uniqueobj_sumkeypts.append(sumkeypts)
        sumkeypts += numkeypts
    pts_vector = torch.cat(pts_vector, dim=0)  # (N, 3)
    pts_uniqueobjindices_vector = torch.cat(pts_uniqueobjindices_vector, dim=0)  # (N,)
    pts_corners_uniqueobjindices_vector = torch.cat(pts_corners_uniqueobjindices_vector, dim=0)  # (Ncor,)
    pts_keyptindices_mask = torch.cat(pts_keyptindices_mask, dim=0)  # (N,)

    dict_frame_observation = {}  # for each frame, the observations in it.
    dict_frame_observation_pts_indices = {}
    dict_frame_observation_object_indices = {}  # for each observ, the index of the object it belongs to.
    dict_frame_observation_keypt_indices = {}
    for indexEdge in range(targetObjectIndices.shape[0]):
        if targetObjectIndices[indexEdge] == -1:
            continue
        # append observations
        indexDestFrame = obj_jj[indexEdge].item()
        if obj_jj[indexEdge].item() == obj_ii[indexEdge].item():
            continue  # Same object
        if indexDestFrame not in dict_frame_observation:
            dict_frame_observation[indexDestFrame] = []
            dict_frame_observation_pts_indices[indexDestFrame] = []
            dict_frame_observation_object_indices[indexDestFrame] = []
            dict_frame_observation_keypt_indices[indexDestFrame] = []
        indexDestObject = targetObjectIndices[indexEdge].item()
        numkeypts_src = objlabel_numkeypts[obj_labels[indexDestObject]].item()
        for indexKeypt in range(numkeypts_src):
            # TODO: obj_2dkeypts need to go to the normalized plane.
            normalized_2dkeypts = obj_2dkeypts[indexDestObject][indexKeypt][None, :].clone()
            normalized_2dkeypts[:, [0, 2]] = (normalized_2dkeypts[:, [0, 2]] - intrinsics[2]) / intrinsics[0]
            normalized_2dkeypts[:, 1] = (normalized_2dkeypts[:, 1] - intrinsics[3]) / intrinsics[1]
            dict_frame_observation[indexDestFrame].append(normalized_2dkeypts)
            dict_frame_observation_pts_indices[indexDestFrame].append((uniqueobj_sumkeypts[obj_uniqueobj_indices[indexDestObject].item()] + torch.as_tensor(indexKeypt, device=device)).unsqueeze(0))
            dict_frame_observation_object_indices[indexDestFrame].append(torch.as_tensor(indexDestObject, device=device))
            dict_frame_observation_keypt_indices[indexDestFrame].append(torch.as_tensor(indexKeypt, device=device))

    all_related_frames_indices = list(dict_frame_observation.keys())
    all_related_frames_indices.sort()
    observations = []
    for indexFrame in all_related_frames_indices:
        one_frame_observs = (
            torch.cat(dict_frame_observation[indexFrame], dim=0),  # (NumObservations, 3)
            torch.cat(dict_frame_observation_pts_indices[indexFrame], dim=0)  # (NumObservations,)
        )
        observations.append(one_frame_observs)

    # prepare frame poses.
    pose_vector = []
    pose_indices = copy.deepcopy(all_related_frames_indices)
    fixed_poses_mask = []
    device = poses.device
    for indexFrame in pose_indices:
        if indexFrame >= t0 and indexFrame < t1:
            fixed_poses_mask.append(0)
        else:
            fixed_poses_mask.append(1)
        q_i_inv = poses[indexFrame][3:].clone()
        q_i_inv[:3] *= -1
        # inverse the transform to: cam <- world
        q_i_inv_ = torch.Tensor([q_i_inv[-1], q_i_inv[0], q_i_inv[1], q_i_inv[2]]).float().to(poses.device)
        rm_i = quaternion_to_rotation_matrix(q_i_inv_)
        t_i_inv = poses[indexFrame][:3].clone()
        t_i_inv = -rm_i @ t_i_inv
        raa_i = rotation_matrix_to_angle_axis(rm_i)
        pose_vector.append(torch.cat([raa_i, t_i_inv], dim=0).unsqueeze(0))
    pose_vector = torch.cat(pose_vector, dim=0)  # (NumFrames, 6)
    pose_indices = torch.Tensor(pose_indices).to(device=device)  # (NumFrames,)
    fixed_poses_mask = torch.Tensor(fixed_poses_mask).to(device=device, dtype=torch.bool)  # do not update the fixed frame poses.

    # start optimization
    pts_vector_hat, pose_vector_hat, delta_x, delta_theta, all_residuals = lm_optimize(
        calibrated_stereo_residuals,
        pts_vector,
        pose_vector,
        observations,
        stereo_baseline=intrinsics[-1].item(),
        intrinsics=intrinsics.cpu().numpy(),
        dtype=dtype,
        L_0 = L_0,
        num_steps=num_steps)

    for indexR, residuals in enumerate(all_residuals):
        print(f"final residuals ({indexR}-th frame) is\n", residuals)
        print("--------------------")

    # update points
    delta_x_foreachobj = delta_x[pts_keyptindices_mask]
    selected_obj_indices = pts_uniqueobjindices_vector[pts_keyptindices_mask]
    lookup = torch.empty_like(delta_x_foreachobj)
    lookup[selected_obj_indices] = delta_x_foreachobj
    delta_x_final = lookup[pts_uniqueobjindices_vector]  # all non-principla keypoints align with principal keypoints.
    delta_x_corners_final = lookup[pts_corners_uniqueobjindices_vector]

    offsets = torch.zeros_like(pts_uniqueobjindices_vector)
    counts = torch.zeros(pts_uniqueobjindices_vector.max() + 1, device=pts_uniqueobjindices_vector.device)
    for i in range(pts_uniqueobjindices_vector.size(0)):
        val = pts_uniqueobjindices_vector[i]
        offsets[i] = counts[val]
        counts[val] += 1
    offsets_corners = torch.zeros_like(pts_corners_uniqueobjindices_vector)
    counts = torch.zeros(pts_corners_uniqueobjindices_vector.max() + 1, device=pts_corners_uniqueobjindices_vector.device)
    for i in range(pts_corners_uniqueobjindices_vector.size(0)):
        val = pts_corners_uniqueobjindices_vector[i]
        offsets_corners[i] = counts[val]
        counts[val] += 1
    uniqueobj_3dkeypts_inworld[pts_uniqueobjindices_vector, offsets] += delta_x_final
    uniqueobj_3dcorners_inworld[pts_corners_uniqueobjindices_vector, offsets_corners] += delta_x_corners_final

    # update poses
    delta_theta_quaternion = torch.zeros(delta_theta.shape[0], 7).to(device=delta_theta.device, dtype=delta_theta.dtype)
    Rs = axis_angle_to_rotation_matrix(pose_vector_hat[:, :3]).transpose(-1, -2)
    delta_theta_quaternion[:, :3] = -torch.bmm(Rs, pose_vector_hat[:, 3:].unsqueeze(-1)).squeeze(-1)
    Qs = rotation_matrix_to_quaternion(Rs)
    Qs = torch.cat([Qs[:, 1:], Qs[:, :1]], dim=1)
    delta_theta_quaternion[:, 3:] = Qs
    poses[pose_indices.to(torch.long)[~fixed_poses_mask]] = delta_theta_quaternion[~fixed_poses_mask]

    return uniqueobj_3dkeypts_inworld, uniqueobj_3dcorners_inworld, poses
