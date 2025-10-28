import trimesh
import mcubes
from scene import Scene


def show_meshes():
    scene_num = 1
    scene_dir = f'scenes/scene_{scene_num}'
    configuration = [('front', 'top', 0.7), ('back', 'top', 0.7), ('top', 'front', 0.7), ('bottom', 'back', 0.7)]
    scene = Scene(scene_dir=scene_dir, depth_configurations=configuration)
    scene.generate_point_clouds()
    meshes = list(scene.convert_to_meshes().values())

    assert len(meshes) > 0, 'No meshes found'

    # export_file = f'mesh--{scene_dir.replace('/', '_')}--{scene.configurations_str}.obj'
    export_file = f'mesh-scene{scene_num}-conf__{scene.configurations_str}.obj'

    trimesh_scene = trimesh.Scene(meshes)
    trimesh_scene.export(file_obj=export_file, file_type='obj')


def show_point_clouds():
    scene_num = 1
    scene_dir = f'scenes/scene_{scene_num}'
    scene = Scene(scene_dir=scene_dir)
    scene.generate_point_clouds()
    # point_clouds = scene.available_point_clouds()
    point_clouds = scene.masked_point_clouds()
    trimes_point_clouds = [trimesh.PointCloud(cloud) for cloud in point_clouds.values()]

    export_file = f'cloud--{scene_dir.replace('/', '_')}--{scene.configurations_str}.obj'
    trimesh_scene = trimesh.Scene(trimes_point_clouds)
    trimesh_scene.export(file_obj=export_file, file_type='obj')


def voxel_voting():
    scene_num = 1
    min_vote_required = 3
    positions = ['front', 'top', 'back', 'bottom']
    scene_dir = f'scenes/scene_{scene_num}'
    scene = Scene(scene_dir=scene_dir)
    cloud, volume_mask = scene.create_voxel_volume(positions=positions, min_vote_required=min_vote_required)
    volume_mask_smooth = mcubes.smooth(volume_mask.numpy(), method='gaussian')
    vertices, triangles = mcubes.marching_cubes(volume_mask_smooth, 0)
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    mesh.apply_translation(-mesh.centroid)
    export_file_name = f'volume-scene{scene_num}-cam{len(positions)}-minvote{min_vote_required}-cams__{'_'.join(positions)}.obj'
    mesh.export(file_obj=export_file_name, file_type='obj')


def main():
    # show_meshes()
    # show_point_clouds()
    voxel_voting()


if __name__ == '__main__':
    main()
