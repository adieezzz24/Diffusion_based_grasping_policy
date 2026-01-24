import torch
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.sensors import Camera
from omni.isaac.lab.scene import InteractiveScene
from omni.isaac.lab.utils.assets import check_file_path
from omni.isaac.core.utils.stage import add_reference_to_stage

class UR5GraspEnv:
    def __init__(self, headless=False):
        # Note: AppLauncher will be called in the script importing this class.
        # initialize the interactive scene here.
        self.scene = InteractiveScene(num_envs=1)
        self._setup_robot()
        self._setup_objects()
        self._setup_camera()
        
    def _setup_robot(self):
        """Initializes the UR5 robot."""
        from omni.isaac.lab.assets import ArticulationCfg
        # Using a standard UR5 config, for now 
        # need to point to a specific USD file or use Isaac Lab's registry
        robot_cfg = ArticulationCfg(
            prim_path="/World/Robot",
            spawn=some_default_spawn_config, # Placeholder for actual spawn config
        )
        self.robot = Articulation(cfg=robot_cfg)
        self.scene.register_articulation(self.robot)

    def _setup_objects(self):
        """Initializes objects to grasp (Cube/Sphere)."""
        from omni.isaac.lab.assets import RigidObjectCfg
        # Create a simple cube to grasp
        cube_cfg = RigidObjectCfg(
            prim_path="/World/TargetCube",
            spawn=some_default_cube_spawn_config, # Placeholder
        )
        self.cube = RigidObject(cfg=cube_cfg)
        self.scene.register_rigid_object(self.cube)

    def _setup_camera(self):
        """Initializes the RGB camera attached to the wrist or world."""
        from omni.isaac.lab.sensors import CameraCfg
        # Camera mounted on the robot wrist or fixed in environment
        camera_cfg = CameraCfg(
            prim_path="/World/Camera",
            offset=CameraCfg.OffsetCfg(pos=(0.5, 0.0, 0.5), rot=(0.707, 0.0, 0.707, 0.0)),
            update_period=0.1,
            height=128, width=128,
            data_types=["rgb", "distance_to_image_plane"],
        )
        self.camera = Camera(cfg=camera_cfg)
        self.scene.register_sensor(self.camera)

    def reset(self):
        """Resets the robot and randomizes object position."""
        self.scene.reset()
        # Randomize cube position within reach
        random_pos = (torch.rand(3) * 0.2) + torch.tensor([0.5, 0.0, 0.1])
        self.cube.write_root_state_to_sim(torch.cat([random_pos, torch.zeros(3), torch.tensor([1.0]), torch.zeros(3)]))

    def step(self):
        """Steps the physics simulation."""
        self.scene.update(dt=0.01)

    def get_observation(self):
        """Returns current image and joint state."""
        # Get RGB image [Batch, Height, Width, Channels]
        rgb_data = self.camera.data.output["rgb"][0].clone() 
        # Get Joint Positions
        joint_pos = self.robot.data.joint_pos.clone()
        return rgb_data, joint_pos
