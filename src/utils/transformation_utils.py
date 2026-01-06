import torch
import numpy as np

from src.mpm_core.warp import SVD

def flatten_and_normalize(tensor, n_particles):
    # normalize tensor to [-1, 1]
    flattened_tensor = tensor.reshape(n_particles, -1)
    min_vals = torch.min(flattened_tensor, dim=0, keepdim=True)
    max_vals = torch.max(flattened_tensor, dim=0, keepdim=True)
    flattened_tensor = (flattened_tensor - min_vals.values) / ( max_vals.values - min_vals.values + 1e-6)
    flattened_tensor = 2 * flattened_tensor - 1
    return flattened_tensor

def angle2vector(angle):
    angle = torch.tensor([angle / 180.0 * 3.1415926], device="cuda")
    return torch.tensor([torch.cos(angle), torch.sin(angle), 0], device="cuda")

def transform2origin(position_tensor, factor=0.95):
    min_pos = torch.min(position_tensor, 0)[0]
    max_pos = torch.max(position_tensor, 0)[0]
    max_diff = torch.max(max_pos - min_pos)
    original_mean_pos = (min_pos + max_pos) / 2.0
    scale = factor / max_diff # set to 0.95 to avoid numerical issue on the boundary
    original_mean_pos = original_mean_pos.to(device="cuda")
    scale = scale.to(device="cuda")
    new_position_tensor = (position_tensor - original_mean_pos) * scale
    return new_position_tensor, scale, original_mean_pos


def undotransform2origin(position_tensor, scale, original_mean_pos):
    return original_mean_pos + position_tensor / scale


def generate_rotation_matrix(degree, axis):
    cos_theta = torch.cos(degree / 180.0 * 3.1415926)
    sin_theta = torch.sin(degree / 180.0 * 3.1415926)
    if axis == 0:
        rotation_matrix = torch.tensor(
            [[1, 0, 0], [0, cos_theta, -sin_theta], [0, sin_theta, cos_theta]]
        )
    elif axis == 1:
        rotation_matrix = torch.tensor(
            [[cos_theta, 0, sin_theta], [0, 1, 0], [-sin_theta, 0, cos_theta]]
        )
    elif axis == 2:
        rotation_matrix = torch.tensor(
            [[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]]
        )
    else:
        raise ValueError("Invalid axis selection")
    return rotation_matrix.cuda()


def generate_rotation_matrices(degrees, axises):
    assert len(degrees) == len(axises)

    matrices = []

    for i in range(len(degrees)):
        matrices.append(generate_rotation_matrix(degrees[i], axises[i]))

    return matrices


def apply_rotation(position_tensor, rotation_matrix):
    rotated = torch.mm(position_tensor, rotation_matrix.T)
    return rotated


def apply_cov_rotation(cov_tensor, rotation_matrix):
    rotated = torch.matmul(cov_tensor, rotation_matrix.T)
    rotated = torch.matmul(rotation_matrix, rotated)
    return rotated


def get_mat_from_upper(upper_mat):
    upper_mat = upper_mat.reshape(-1, 6)
    mat = torch.zeros((upper_mat.shape[0], 9), device="cuda")
    mat[:, :3] = upper_mat[:, :3]
    mat[:, 3] = upper_mat[:, 1]
    mat[:, 4] = upper_mat[:, 3]
    mat[:, 5] = upper_mat[:, 4]
    mat[:, 6] = upper_mat[:, 2]
    mat[:, 7] = upper_mat[:, 4]
    mat[:, 8] = upper_mat[:, 5]

    return mat.view(-1, 3, 3)


def get_uppder_from_mat(mat):
    mat = mat.view(-1, 9)
    upper_mat = torch.zeros((mat.shape[0], 6), device="cuda")
    upper_mat[:, :3] = mat[:, :3]
    upper_mat[:, 3] = mat[:, 4]
    upper_mat[:, 4] = mat[:, 5]
    upper_mat[:, 5] = mat[:, 8]

    return upper_mat


def apply_rotations(position_tensor, rotation_matrices):
    for i in range(len(rotation_matrices)):
        position_tensor = apply_rotation(position_tensor, rotation_matrices[i])
    return position_tensor


def apply_cov_rotations(upper_cov_tensor, rotation_matrices):
    cov_tensor = get_mat_from_upper(upper_cov_tensor)
    for i in range(len(rotation_matrices)):
        cov_tensor = apply_cov_rotation(cov_tensor, rotation_matrices[i])
    return get_uppder_from_mat(cov_tensor)

def shift2center111(position_tensor):
    tensor111 = torch.tensor([1.0, 1.0, 1.0], device="cuda")
    return position_tensor + tensor111


def undoshift2center111(position_tensor):
    tensor111 = torch.tensor([1.0, 1.0, 1.0], device="cuda")
    return position_tensor - tensor111


def shift2center05(position_tensor):
    tensor05 = torch.tensor([0.5, 0.5, 0.5], device="cuda")
    return position_tensor + tensor05


def undoshift2center05(position_tensor):
    tensor05 = torch.tensor([0.5, 0.5, 0.5], device="cuda")
    return position_tensor - tensor05


def apply_inverse_rotation(position_tensor, rotation_matrix):
    rotated = torch.mm(position_tensor, rotation_matrix)
    return rotated


def apply_inverse_rotations(position_tensor, rotation_matrices):
    for i in range(len(rotation_matrices)):
        R = rotation_matrices[len(rotation_matrices) - 1 - i]
        position_tensor = apply_inverse_rotation(position_tensor, R)
    return position_tensor


def apply_inverse_cov_rotations(upper_cov_tensor, rotation_matrices):
    cov_tensor = get_mat_from_upper(upper_cov_tensor)
    for i in range(len(rotation_matrices)):
        R = rotation_matrices[len(rotation_matrices) - 1 - i]
        cov_tensor = apply_cov_rotation(cov_tensor, R.T)
    return get_uppder_from_mat(cov_tensor)


# input must be (n,3) tensor on cuda
def undo_all_transforms(input, rotation_matrices, scale_origin, original_mean_pos):
    return apply_inverse_rotations(
        undotransform2origin(
            undoshift2center05(input), scale_origin, original_mean_pos
        ),
        rotation_matrices,
    )


# supply vertical vector in world space
def generate_local_coord(vertical_vector):
    vertical_vector = vertical_vector / np.linalg.norm(vertical_vector)
    horizontal_1 = np.array([1, 1, 1])
    if np.abs(np.dot(horizontal_1, vertical_vector)) < 0.01:
        horizontal_1 = np.array([0.72, 0.37, -0.67])
    # gram schimit
    horizontal_1 = (
        horizontal_1 - np.dot(horizontal_1, vertical_vector) * vertical_vector
    )
    horizontal_1 = horizontal_1 / np.linalg.norm(horizontal_1)
    horizontal_2 = np.cross(horizontal_1, vertical_vector)

    return vertical_vector, horizontal_1, horizontal_2


def get_center_view_worldspace_and_observant_coordinate(
    mpm_space_viewpoint_center,
    mpm_space_vertical_upward_axis,
    rotation_matrices,
    scale_origin,
    original_mean_pos,
):
    viewpoint_center_worldspace = undo_all_transforms(
        mpm_space_viewpoint_center, rotation_matrices, scale_origin, original_mean_pos
    )
    mpm_space_up = mpm_space_vertical_upward_axis + mpm_space_viewpoint_center
    worldspace_up = undo_all_transforms(
        mpm_space_up, rotation_matrices, scale_origin, original_mean_pos
    )
    world_space_vertical_axis = worldspace_up - viewpoint_center_worldspace
    viewpoint_center_worldspace = np.squeeze(
        viewpoint_center_worldspace.clone().detach().cpu().numpy(), 0
    )
    vertical, h1, h2 = generate_local_coord(
        np.squeeze(world_space_vertical_axis.clone().detach().cpu().numpy(), 0)
    )
    observant_coordinates = np.column_stack((h1, h2, vertical))

    return viewpoint_center_worldspace, observant_coordinates

def compute_cov_from_F(init_cov, F):
    # compute temp cov from init cov (temp_conv=F*init_cov*F^T)
    cov_tensor = get_mat_from_upper(init_cov)
    cov_tensor = torch.matmul(F, torch.matmul(cov_tensor, torch.transpose(F, 1, 2)))
    return get_uppder_from_mat(cov_tensor)

def compute_R_from_F(F):
    svd = SVD()
    U, _, V = svd(F)
    U_det = torch.det(U)
    V_det = torch.det(V)
    U[U_det < 0.0, 2] = -U[U_det < 0.0, 2]
    V[V_det < 0.0, 2] = -V[V_det < 0.0, 2]
    R = torch.matmul(U, V)
    return R.transpose(1, 2)

def get_mpm_gaussian_params(
    pos, cov, shs, opacity,
    F,
    unselected_params,
    rotation_matrices, scale_origin, original_mean_pos
):
    render_pos = undo_all_transforms(
        pos, rotation_matrices, scale_origin, original_mean_pos
    )
    
    render_cov = compute_cov_from_F(cov, F)
    render_cov = render_cov / (scale_origin * scale_origin)
    render_cov = apply_inverse_cov_rotations(render_cov, rotation_matrices)
    
    render_shs = shs
    render_opacity = opacity
    
    # concat unselected params
    unselected_pos = unselected_params['pos']
    unselected_cov = unselected_params['cov']
    unselected_opacity = unselected_params['opacity']
    unselected_shs = unselected_params['shs']
    if unselected_pos is not None:
        render_pos = torch.cat((render_pos, unselected_pos), dim=0)
        render_cov = torch.cat((render_cov, unselected_cov), dim=0)
        render_shs = torch.cat((shs, unselected_shs), dim=0)
        render_opacity = torch.cat((opacity, unselected_opacity), dim=0)
    
    # get rotation for color precomp
    render_rot = compute_R_from_F(F)
    
    return render_pos, render_cov, render_shs, render_opacity, render_rot

def filter_cov(cov, threshold=1e-4):
    # calculate the eigenvalues of the covariance matrix
    eig_values = torch.linalg.eigvalsh(cov)
    # filter the eigenvalues
    max_eig_values = torch.max(eig_values, dim=1).values # n
    valid = max_eig_values < threshold
    return valid