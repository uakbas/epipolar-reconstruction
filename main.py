import trimesh
from scene import Scene


def main():
    scene = Scene(scene_dir='scenes/scene_1')
    scene.generate_point_clouds()
    meshes = scene.convert_to_meshes()
    meshes = [mesh for mesh in list(meshes.values()) if mesh is not None]

    if len(meshes) == 0:
        print('No mesh generated.')
        return None

    trimesh_scene = trimesh.Scene(meshes)
    trimesh_scene.export(file_obj='fish_mesh.obj', file_type='obj')


if __name__ == '__main__':
    main()
