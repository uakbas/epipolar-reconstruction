import trimesh
from scene import Scene


def show_meshes():
    scene = Scene(scene_dir='scenes/scene_1')
    scene.generate_point_clouds()
    meshes = list(scene.convert_to_meshes().values())

    if len(meshes) == 0:
        print('No mesh generated.')
        return

    ax = trimesh.creation.axis(origin_size=1, axis_length=100)
    meshes.append(ax)

    trimesh_scene = trimesh.Scene(meshes)
    trimesh_scene.export(file_obj='fish_mesh.obj', file_type='obj')


def show_point_clouds():
    scene = Scene(scene_dir='scenes/scene_1')
    scene.generate_point_clouds()
    # point_clouds = scene.available_point_clouds()
    point_clouds = scene.masked_point_clouds()
    trimes_point_clouds = [trimesh.PointCloud(cloud) for cloud in point_clouds.values()]
    trimesh_scene = trimesh.Scene(trimes_point_clouds)
    trimesh_scene.export(file_obj='fish_cloud_masked.obj', file_type='obj')


def main():
    show_meshes()
    # show_point_clouds()


if __name__ == '__main__':
    main()
