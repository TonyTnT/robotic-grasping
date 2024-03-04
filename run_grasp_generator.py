from inference.grasp_generator import GraspGenerator

if __name__ == '__main__':
    generator = GraspGenerator(
        cam_id=130322270731,
        saved_model_path='trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch16/epoch_20_iou_0.97',
        visualize=True
    )
    generator.load_model()
    generator.run()
