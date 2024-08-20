import trimesh
from scene import Scene


def show_meshes():
    scene_num = 1
    scene_dir = f'scenes/scene_{scene_num}'
    scene = Scene(scene_dir=scene_dir)
    scene.generate_point_clouds()
    meshes = list(scene.convert_to_meshes().values())

    assert len(meshes) > 0, 'No meshes found'

    export_file = f'mesh--{scene_dir.replace('/', '_')}--{scene.configurations_str}.obj'

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
    scene_dir = f'scenes/scene_{scene_num}'
    scene = Scene(scene_dir=scene_dir)
    cloud = scene.create_point_cloud_by_voxel_voting()
    t_cloud = trimesh.PointCloud(cloud)
    t_cloud.export(file_obj='voxel_voting.obj')


def main():
    # show_meshes()
    # show_point_clouds()
    voxel_voting()


if __name__ == '__main__':
    main()
