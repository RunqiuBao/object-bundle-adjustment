import numpy as np


def DebugObjectTargetAssignment(
    obj_poses,
    objlabel_numkeypts,
    obj_labels,
    uniqueobj_3dkeypts_inworld,
    uniqueobj_3dcorners_inworld,
    obj_uniqueobj_indices,
    obj_2dkeypts,
    obj_2dcorners,
    obj_ii,
    obj_jj,
    obj_kk,
    targetObjectIndices,
    cameraParameters,  # keys required: cameraKK (3x3), imageShape (h, w),
    savePath
) -> None:
    """
    Given observ edges, project source objects and compare with the targets.
    Draw all the target frames, projections and target objects should be indexed and connected by line.
    """
    import numpy as np
    import os
    import cv2


    def ConvertQuatToRotMat(q):
        x, y, z, w = q
        return np.array([
            [1 - 2*y**2 - 2*z**2,   2*x*y - 2*z*w,       2*x*z + 2*y*w],
            [2*x*y + 2*z*w,         1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w,         2*y*z + 2*x*w,       1 - 2*x**2 - 2*y**2]
        ])


    def ConvertPoseToTransform(pose: np.ndarray):
        assert pose.shape[0] == 7
        transform = np.eye(4)
        transform[:3, 3] = pose[:3]
        transform[:3, :3] = ConvertQuatToRotMat(pose[-4:])
        return transform

    os.makedirs(savePath, exist_ok=True)

    targetFrames = {}  # frameIndex: debugMaskImage
    cameraKK = cameraParameters["cameraKK"]
    imageShape = cameraParameters["imageShape"]
    largestRreprojectionDistance = 0.0
    indexLargestRreprojectionDistance = -1
    for indexEdge, indexTargetFrame in enumerate(obj_jj):
        if targetObjectIndices[indexEdge] == -1:
            continue
        if targetObjectIndices[indexEdge] == obj_kk[indexEdge]:
            continue
        if indexTargetFrame not in targetFrames:
            targetFrames[indexTargetFrame] = np.zeros((*imageShape, 3), dtype="uint8")

        Tdw = np.linalg.inv(ConvertPoseToTransform(obj_poses[obj_jj[indexEdge]]))
        srcObjectCornersInDest = uniqueobj_3dcorners_inworld[obj_uniqueobj_indices[obj_kk[indexEdge]]] @ Tdw[:3, :3].T + Tdw[:3, 3]
        projectedCorners = srcObjectCornersInDest @ cameraKK.T
        projectedCorners = projectedCorners[:, :2] / projectedCorners[:, 2][:, np.newaxis]
        projectedCorners = np.round(projectedCorners).astype("int32")
        # Extract convex hull for the projected corners
        hull = cv2.convexHull(projectedCorners)
        # Draw the convex hull
        color = tuple(np.random.randint(0, 256, size=3).tolist())  # Generate a random color
        cv2.polylines(targetFrames[indexTargetFrame], [hull], isClosed=True, color=color, thickness=1)
        # Write ID at the center of the polygon
        M = cv2.moments(hull)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            xText = np.min(hull.squeeze(1), axis=0)[0] - 5
            yText = np.min(hull.squeeze(1), axis=0)[1] - 5
            cv2.putText(targetFrames[indexTargetFrame], str(obj_kk[indexEdge]), (xText, yText), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        # draw the projected keypoints
        for indexKeypt in range(objlabel_numkeypts[int(obj_labels[obj_kk[indexEdge]])]):
            keypoint3d = uniqueobj_3dkeypts_inworld[obj_uniqueobj_indices[obj_kk[indexEdge]]][indexKeypt]
            keypoint3dInDest = keypoint3d @ Tdw[:3, :3].T + Tdw[:3, 3]
            projectedKeypoint = keypoint3dInDest @ cameraKK.T
            projectedKeypoint = projectedKeypoint[:2] / projectedKeypoint[2]
            projectedKeypoint = np.round(projectedKeypoint).astype("int")
            cv2.circle(targetFrames[indexTargetFrame], (projectedKeypoint[0], projectedKeypoint[1]), radius=3, color=(255, 255, 255), thickness=1)
        print("indexTargetFrame: {}, indexSrcObject: {}, indexDestObject: {}, indexUniqueObject: {}, projectedKeypoint: {}".format(indexTargetFrame, obj_kk[indexEdge], targetObjectIndices[indexEdge], obj_uniqueobj_indices[targetObjectIndices[indexEdge]], (cX, cY)))

        # Draw the target object mask
        color = tuple(int(255 - c) for c in color)  # Compute the opposite color
        targetCorners = obj_2dcorners[targetObjectIndices[indexEdge]].astype("int32")
        hull2 = cv2.convexHull(targetCorners)
        cv2.fillPoly(targetFrames[indexTargetFrame], [hull2], color=color)
        M2 = cv2.moments(hull2)
        if M2["m00"] != 0:
            cX2 = int(M2["m10"] / M2["m00"])
            cY2 = int(M2["m01"] / M2["m00"])
            cv2.putText(targetFrames[indexTargetFrame], str(targetObjectIndices[indexEdge]), (cX2, cY2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        for indexKeypt in range(objlabel_numkeypts[int(obj_labels[targetObjectIndices[indexEdge]])]):
            keypoint2d = obj_2dkeypts[targetObjectIndices[indexEdge]][indexKeypt]
            keypoint2d = np.round(keypoint2d).astype("int")
            cv2.circle(targetFrames[indexTargetFrame], (keypoint2d[0], keypoint2d[1]), radius=3, color=(255, 255, 255), thickness=-1)
            # if indexKeypt > 0 and np.linalg.norm(np.mean(hull2.squeeze(1), axis=0) - projectedKeypoint) > 20:

        # Draw lines connecting projected center to target center
        cv2.line(targetFrames[indexTargetFrame], (cX, cY), (cX2, cY2), color=(255, 255, 255), thickness=1)
        dist = np.sqrt((cX - cX2)**2 + (cY - cY2)**2)
        if dist > largestRreprojectionDistance:
            largestRreprojectionDistance = dist
            indexLargestRreprojectionDistance = indexEdge
        cv2.putText(targetFrames[indexTargetFrame], f"{dist:.1f}", (int((cX + cX2) / 2.0), int((cY + cY2) / 2.0)), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255), 1, cv2.LINE_AA)
        print("targetKeypoint: {}".format((cX2, cY2)))
        print("--------------------")

    print(f"Largest reprojection distance: {largestRreprojectionDistance} at index {indexLargestRreprojectionDistance}")
    if largestRreprojectionDistance > 10:
        return True
    
    return False

# isBug = DebugObjectTargetAssignment(
#     poses.cpu().numpy(),
#     objlabel_numkeypts.cpu().numpy(),
#     obj_labels.cpu().numpy(),
#     uniqueobj_3dkeypts_inworld.cpu().numpy(),
#     uniqueobj_3dcorners_inworld.cpu().numpy(),
#     obj_uniqueobj_indices.cpu().numpy(),
#     obj_2dkeypts.cpu().numpy(),
#     obj_2dcorners.cpu().numpy(),
#     obj_ii.cpu().numpy(),
#     obj_jj.cpu().numpy(),
#     obj_kk.cpu().numpy(),
#     targetObjectIndices.cpu().numpy(),
#     {
#         "cameraKK": np.array([[intrinsics[0].item(), 0, intrinsics[2].item()], [0, intrinsics[1].item(), intrinsics[3].item()], [0, 0, 1]]),
#         "imageShape": (480, 640)
#     },
#     "/root/code/DEVO/debugImages/"
# )


def compare_reproj(
    indexTargetFrame,
    poses,
    indexTargetObject,
    obj_uniqueobj_indices,
    obj_2dkeypts,
    uniqueobj_3dkeypts_inworld,
    intrinsics
):
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    pose = poses[indexTargetFrame].cpu().numpy()
    mpose = np.eye(4)
    mpose[:3, 3] = pose[:3]
    r = R.from_quat(pose[3:])
    mpose[:3, :3] = r.as_matrix()
    Tfw = np.linalg.inv(mpose)
    points3d_inworld = uniqueobj_3dkeypts_inworld[obj_uniqueobj_indices[indexTargetObject]].cpu().numpy()[:2]
    points3d_inframe = points3d_inworld @ Tfw[:3, :3].T + Tfw[:3, 3]
    points2d = points3d_inframe @ np.array([[intrinsics[0].item(), 0, intrinsics[2].item()], [0, intrinsics[1].item(), intrinsics[3].item()], [0, 0, 1]]).T
    points2d /= np.repeat(points2d[:, -1][:, None], 3, axis=1)
    Trl = np.eye(4)
    Trl[0, 3] = -intrinsics[-1].item()
    Trw = Trl @ Tfw
    points3d_inright = points3d_inworld @ Trw[:3, :3].T + Trw[:3, 3]
    points2d_r = points3d_inright @ np.array([[intrinsics[0].item(), 0, intrinsics[2].item()], [0, intrinsics[1].item(), intrinsics[3].item()], [0, 0, 1]]).T
    points2d_r /= np.repeat(points2d_r[:, -1][:, None], 3, axis=1)
    points2d = np.concatenate([points2d[:, :2], points2d_r[:, 0][:, None]], axis=-1)
    print("projected points2d:\n{}".format(points2d))
    keypts2d = obj_2dkeypts[indexTargetObject].cpu().numpy()[:2]
    print("keypts2d:\n{}".format(keypts2d))
    print("residual:\n{}".format(keypts2d - points2d))
