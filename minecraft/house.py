from random import randint

from gdpc import geometry
from gdpc.block import Block
import torch

from utils import Utils
from minecraft_utils import MinecraftUtils



class House: 
    
    # house features (4x6 features = 24 features total)
    FARM_ROOF = 'farm_roof'
    FARM_WALL = 'farm_wall'
    FARM_DOOR = 'farm_door'
    FARM_WINDOW = 'farm_window'
    FARM_PORCH = 'farm_porch_deco1'
    FARM_CROP = 'farm_crop_deco2'
    MEDIEVAL_ROOF = 'medieval_roof'
    MEDIEVAL_WALL = 'medieval_wall'
    MEDIEVAL_DOOR = 'medieval_door'
    MEDIEVAL_WINDOW = 'medieval_window'
    MEDIEVAL_CHIMNEY = 'medieval_chimney_deco1'
    MEDIEVAL_STABLE = 'medieval_stable_deco2'
    MODERN_ROOF = 'modern_roof'
    MODERN_WALL = 'modern_wall'
    MODERN_DOOR = 'modern_door'
    MODERN_WINDOW = 'modern_window'
    MODERN_GATE = 'modern_gate_deco1'
    MODERN_POOL = 'modern_pool_deco2'
    HAUNTED_ROOF = 'haunted_roof'
    HAUNTED_WALL = 'haunted_wall'
    HAUNTED_DOOR = 'haunted_door'
    HAUNTED_WINDOW = 'haunted_window'
    HAUNTED_ENTRANCE = 'haunted_spider_webs_deco1'
    HAUNTED_GRAVE_YARD = 'haunted_grave_yard_deco2'
    
    def __init__(self):
        self.props = []
        self.actions = {
            self.FARM_ROOF: self.add_barn_roof, 
            self.FARM_WALL: self.add_brick_walls,
            self.FARM_DOOR: self.add_oak_door,
            self.FARM_WINDOW: self.add_farm_windows,
            self.FARM_PORCH: self.add_front_porch,
            self.FARM_CROP: self.add_crop,
            self.MEDIEVAL_ROOF: self.add_thatched_roof,
            self.MEDIEVAL_WALL: self.add_stone_walls,
            self.MEDIEVAL_DOOR: self.add_dark_wood_door,
            self.MEDIEVAL_WINDOW: self.add_shutter_windows,
            self.MEDIEVAL_CHIMNEY: self.add_stone_chimney,
            self.MEDIEVAL_STABLE: self.add_stable,
            self.MODERN_ROOF: self.add_flat_roof,
            self.MODERN_WALL: self.add_concrete_walls,
            self.MODERN_DOOR: self.add_iron_door,
            self.MODERN_WINDOW: self.add_floor_to_ceiling_windows,
            self.MODERN_GATE: self.add_gate,
            self.MODERN_POOL: self.add_pool,
            self.HAUNTED_ROOF: self.add_broken_roof,
            self.HAUNTED_WALL: self.add_dark_wood_walls,
            self.HAUNTED_DOOR: self.add_damaged_door,
            self.HAUNTED_WINDOW: self.add_destroyed_windows,
            self.HAUNTED_ENTRANCE: self.add_creepy_entrance,
            self.HAUNTED_GRAVE_YARD: self.add_grave_yard,
        }
        
        self.sorted_actions = sorted(self.actions.keys())
        
        # randomize house dimensions for added variety
        self.width = Utils.randint_even(9, 13)
        self.depth = Utils.randint_even(9, 13)
        self.height = Utils.randint_even(5, 8)
        

    def add_property(self, action):
        self.props.append(action)
        
    def sort_properties_key(self, item):
        """
        Must be called when showing houses in Minecraft. 
        
        This is used to ensure that house components are built in order 
        (e.g., walls before windows, or else the windows will not show).
        """
        if 'wall' in item:
            return (0, item)
        elif 'roof' in item:
            return (1, item)
        else: 
            return (2, item)
            
        
    def to_tensor(self):
        """
        Convert house properties to a binary tensor representation for neural networks.
        
        Returns a 24-element tensor where each element is 1 if that property
        exists in the house, 0 if not.
        """
        
        property_flag = []
        for i in self.sorted_actions:
            if i in self.props:
                property_flag.append(1)
            else: 
                property_flag.append(0)
        return torch.tensor(property_flag).float()
    
    
    def copy(self):
        new_house = House()
        new_house.props = self.props.copy()
        return new_house
    
    def show(self, editor, origin):
        try:
            sorted_props = sorted(self.props, key=self.sort_properties_key)
            for prop in sorted_props:
                self.actions[prop](editor, origin)
        except Exception as e:
            print(f"Error showing house: {e}")
    
    
    # ADDITIONS FOR GFLOWNET (HOUSE TYPES)
    # ------------------------------------------------------------
    
    # FARMHOUSE
    # ---------- 
    def add_barn_roof(self, editor, origin):
        """A-frame barn roof with attic window."""
        ground_height = MinecraftUtils.get_ground_height(editor)
        
        # calculate how many layers until left and right sides meet
        layers_needed = (self.width // 2) + 1
        roof_height = min(layers_needed, 8)
        
        if roof_height % 2 == 1:  
            center_layer = roof_height // 2
        else:  
            center_layer = (roof_height // 2) - 1
        
        # build the A-frame rows of stairs
        for layer in range(roof_height):
            y_level = ground_height + self.height + layer 
            
            
            left_x = origin[0] + layer - 1
            right_x = origin[0] + self.width - layer
            
            # place left side stairs
            geometry.placeCuboid(
                editor, 
                (left_x, y_level, origin[2] - 1), 
                (left_x, y_level, origin[2] + self.depth), 
                Block("oak_stairs", {"facing": "east"})
            )
            
            # place right side stairs
            geometry.placeCuboid(
                editor,
                (right_x, y_level, origin[2] - 1),
                (right_x, y_level, origin[2] + self.depth),
                Block("oak_stairs", {"facing": "west"})
            )
            
            # fill the space between stairs
            if left_x + 1 < right_x:
                geometry.placeCuboid(
                    editor,
                    (left_x + 1, y_level, origin[2]),
                    (right_x - 1, y_level, origin[2] + self.depth - 1),
                    Block("red_terracotta")
                )
                
                # add glass panes for attic window
                if layer == center_layer:
                    center_x = (left_x + right_x) // 2
                    editor.placeBlock(
                        (center_x, y_level, origin[2]),
                        Block("glass_pane")
                    )
                    editor.placeBlock(
                        (center_x+1, y_level, origin[2]),
                        Block("glass_pane")
                    )
    
    def add_brick_walls(self, editor, origin):
        """Classic brick walls with potential window/door openings."""
        self._place_hollow_cuboid(editor,
                                (0, 0, 0),
                                (self.width-1, self.height-1, self.depth-1),
                                "red_terracotta",
                                origin,
                                thickness=1)

    
    def add_oak_door(self, editor, origin):
        """Place oak door in the pre-made opening."""
        ground_height = MinecraftUtils.get_ground_height(editor)
        
        door_x = origin[0] + self.width // 2 - 0.5
        door_z = origin[2] 
        
        # remove wall for door opening
        geometry.placeCuboid(editor,
            (door_x, ground_height, door_z),
            (door_x + 1, ground_height, door_z),
            Block("air")
        )
        
        geometry.placeCuboid(editor,
            (door_x, ground_height, door_z),    
            (door_x, ground_height, door_z),    
            Block("oak_door[facing=south, half=lower, hinge=right]")
        )
        geometry.placeCuboid(editor,
            (door_x+1, ground_height, door_z),    
            (door_x+1, ground_height, door_z),    
            Block("oak_door[facing=south, half=lower, hinge=left]")
        )
    
    def add_farm_windows(self, editor, origin):
        """Windows for farmhouses"""
        ground_height = MinecraftUtils.get_ground_height(editor)
        window_y = ground_height + (self.height // 2)  
        
        left_window_x = origin[0] + 2  # 2 blocks from left edge
        right_window_x = origin[0] + self.width - 4  # 2 blocks from right edge 
        
        # only place on front wall 
        window_z = origin[2]  
        
        # remove wall for left window opening
        geometry.placeCuboid(editor,
            (left_window_x, window_y, window_z),
            (left_window_x + 1, window_y + 1, window_z),
            Block("air")
        )
        
        # remove wall for right window opening
        geometry.placeCuboid(editor,
            (right_window_x, window_y, window_z),
            (right_window_x + 1, window_y + 1, window_z),
            Block("air")
        )
        
        # left window
        geometry.placeCuboid(editor,
            (left_window_x, window_y, window_z),
            (left_window_x + 1, window_y + 1, window_z),
            Block("glass_pane")
        )
        
        # right window
        geometry.placeCuboid(editor,
            (right_window_x, window_y, window_z),
            (right_window_x + 1, window_y + 1, window_z),
            Block("glass_pane")
        )
                
                
    
    def add_front_porch(self, editor, origin):
        """farmhouse front porch with fences"""
        porch_depth = 3
        ground_height = MinecraftUtils.get_ground_height(editor)
        geometry.placeCuboid(editor,
            (origin[0], ground_height, origin[2] - porch_depth),
            (origin[0] + self.width -1, ground_height, origin[2]-1),
            Block("oak_planks")
        )
        
        # add fence for porch
        geometry.placeCuboid(editor,
            (origin[0], ground_height+1, origin[2] - porch_depth),
            (origin[0], ground_height+1, origin[2]-1),
            Block("oak_fence")
        )
        geometry.placeCuboid(editor,
            (origin[0], ground_height+1, origin[2] - porch_depth),
            (origin[0]+ self.width -1, ground_height+1, origin[2]-porch_depth),
            Block("oak_fence")
        )
        geometry.placeCuboid(editor,
            (origin[0] + self.width -1, ground_height+1, origin[2] - porch_depth),
            (origin[0] + self.width -1, ground_height+1, origin[2]-1),
            Block("oak_fence")
        )
        
        # remove porch blocks for door opening
        geometry.placeCuboid(editor,
            (origin[0] + self.width // 2 - 2, ground_height, origin[2] - porch_depth),
            (origin[0] + self.width // 2 + 1, ground_height+1, origin[2]-1),
            Block("air")
        )
        
        # add stairs for porch
        geometry.placeCuboid(editor,
            (origin[0] + self.width // 2 - 2, ground_height, origin[2] - porch_depth),
            (origin[0] + self.width // 2 - 2, ground_height, origin[2]-1),
            Block("oak_stairs[facing=west]")
        )
        geometry.placeCuboid(editor,
            (origin[0] + self.width // 2 + 1, ground_height, origin[2] - porch_depth),
            (origin[0] + self.width // 2 + 1, ground_height, origin[2]-1),
            Block("oak_stairs[facing=east]")
        )
    
    def add_crop(self, editor, origin):
        """farmland with crops"""
        farm_side_width = 3
        farm_back_width = 5
        ground_height = MinecraftUtils.get_ground_height(editor)
        
        start_x = origin[0] - 1
        end_x = origin[0] + self.width
        start_z = origin[2]
        end_z = origin[2] + self.depth
        
        # place farmland (so crops can grow on it)
        geometry.placeCuboid(editor,
            (start_x - farm_side_width, ground_height-1, start_z),
            (start_x, ground_height-1, end_z+farm_back_width),
            Block("farmland", {"moisture": 7})
        )
        geometry.placeCuboid(editor,
            (end_x + farm_side_width, ground_height-1, start_z),
            (end_x, ground_height-1, end_z+farm_back_width),
            Block("farmland", {"moisture": 7})
        )
        geometry.placeCuboid(editor,
            (start_x, ground_height-1, end_z+farm_back_width),
            (end_x, ground_height-1, end_z),
            Block("farmland", {"moisture": 7})
        )
        
        # add crop to farmland
        geometry.placeCuboid(editor,
            (start_x - farm_side_width, ground_height, start_z),
            (start_x, ground_height, end_z+farm_back_width),
            Block("wheat", {"age": 7})
        )
        geometry.placeCuboid(editor,
            (end_x + farm_side_width, ground_height, start_z),
            (end_x, ground_height, end_z+farm_back_width),
            Block("wheat", {"age": 7})
        )
        geometry.placeCuboid(editor,
            (start_x, ground_height, end_z+farm_back_width),
            (end_x, ground_height, end_z),
            Block("wheat", {"age": 7})
        )
        
        # add fence around farm
        geometry.placeCuboid(editor,
            (start_x, ground_height, start_z),
            (start_x - farm_side_width, ground_height, start_z),
            Block("oak_fence")
        )
        geometry.placeCuboid(editor,
            (start_x - farm_side_width, ground_height, start_z),
            (start_x - farm_side_width, ground_height, end_z+farm_back_width),
            Block("oak_fence")
        )
        geometry.placeCuboid(editor,
            (start_x-farm_side_width+1, ground_height, end_z+farm_back_width),
            (end_x + farm_side_width, ground_height, end_z+farm_back_width),
            Block("oak_fence")
        )
        geometry.placeCuboid(editor,
            (end_x + farm_side_width, ground_height, start_z),
            (end_x + farm_side_width, ground_height, end_z+farm_back_width),
            Block("oak_fence")
        )
        geometry.placeCuboid(editor,
            (end_x + farm_side_width, ground_height, start_z),
            (end_x, ground_height, start_z),
            Block("oak_fence")
        )
    
    
    # MEDIEVAL
    # --------
    def add_thatched_roof(self, editor, origin):
        """A-frame barn roof with glass pane center."""
        ground_height = MinecraftUtils.get_ground_height(editor)
        
        # calculate how many layers until left and right sides meet
        layers_needed = (self.width // 4) + 1
        roof_height = min(layers_needed, 8)
        
        
        # build the A-frame rows of stairs
        for layer in range(roof_height):
            y_level = ground_height + self.height + layer 
            
            left_x = origin[0] + layer
            right_x = origin[0] + self.width - 1 - layer
            
            # place left side stairs
            geometry.placeCuboid(
                editor, 
                (left_x, y_level, origin[2]), 
                (left_x, y_level, origin[2] + self.depth-1), 
                Block("brown_wool")
            )
            
            # place right side stairs
            geometry.placeCuboid(
                editor,
                (right_x, y_level, origin[2]),
                (right_x, y_level, origin[2] + self.depth-1),
                Block("brown_wool")
            )
            
            # fill the space between stairs
            if left_x + 1 < right_x:
                # add brown wool for the top layer, otherwise use cobblestone
                if layer == roof_height-1: 
                    material = "brown_wool"
                else:
                    material = "cobblestone"
                
                geometry.placeCuboid(
                    editor,
                    (left_x + 1, y_level, origin[2]),
                    (right_x - 1, y_level, origin[2] + self.depth - 1),
                    Block(material)
                )
                

    def add_stone_walls(self, editor, origin):
        """Cobblestone walls for peasant medieval houses"""
        self._place_hollow_cuboid(editor,
                                (0, 0, 0),
                                (self.width-1, self.height-1, self.depth-1),
                                "cobblestone",
                                origin,
                                thickness=1)

    def add_dark_wood_door(self, editor, origin):
        """one dark wood door for peasant medieval houses"""
        ground_height = MinecraftUtils.get_ground_height(editor)
        
        door_x = origin[0] + self.width // 2 - 0.5
        door_z = origin[2]  
        
        # remove wall for door opening
        geometry.placeCuboid(editor,
            (door_x, ground_height, door_z),
            (door_x, ground_height, door_z),
            Block("air")
        )
        
        geometry.placeCuboid(editor,
            (door_x, ground_height, door_z),    
            (door_x, ground_height, door_z),    
            Block("dark_oak_door[facing=south, half=lower, hinge=right]")
        )

    def add_shutter_windows(self, editor, origin):
        """Windows with shutters """
        ground_height = MinecraftUtils.get_ground_height(editor)
        window_y = ground_height + (self.height // 2)  
        
        left_window_x = origin[0] + 2  # 2 blocks from left edge
        right_window_x = origin[0] + self.width - 4  # 2 blocks from right edge
        
        # only place on front wall 
        window_z = origin[2]  
        
        # remove wall for window opening
        geometry.placeCuboid(editor,
            (left_window_x, window_y, window_z),
            (left_window_x + 1, window_y, window_z),
            Block("air")
        )
        geometry.placeCuboid(editor,
            (right_window_x, window_y, window_z),
            (right_window_x + 1, window_y, window_z),
            Block("air")
        )
        
        # add trapdoor for shutters
        geometry.placeCuboid(editor,
            (left_window_x, window_y, window_z),
            (left_window_x + 1, window_y, window_z),
            Block("spruce_trapdoor[facing=south,open=true]")
        )
        geometry.placeCuboid(editor,
            (right_window_x, window_y, window_z),
            (right_window_x + 1, window_y, window_z),
            Block("spruce_trapdoor[facing=south,open=true]")
        )

    def add_stone_chimney(self, editor, origin):
        """Stone chimney on roof of house."""
        chimney_width = 2
        chimney_height_start = origin[1] + self.height
        chimney_height_end = chimney_height_start + 5
        chimney_x_start = origin[0] + self.width // 4
        chimney_z_start = origin[2] + self.depth // 4
        
        
        # chimney structure
        geometry.placeCuboid(editor,
            (chimney_x_start, chimney_height_start, chimney_z_start),
            (chimney_x_start + chimney_width-1, chimney_height_end , chimney_z_start + chimney_width-1),
            Block("cobblestone"),
        )
        
        # smoke effect at top (campfire)
        geometry.placeCuboid(editor,
            (chimney_x_start, chimney_height_end, chimney_z_start),
            (chimney_x_start + chimney_width-1, chimney_height_end, chimney_z_start + chimney_width-1),
            Block("campfire[lit=true]"),
        )

    def add_stable(self, editor, origin):
        start_x = origin[0]
        end_x = origin[0] + self.width - 1
        start_z = origin[2] + self.depth
        start_leanto_height = origin[1] + self.height - 1
        end_leanto_height = start_leanto_height - (self.height)//2 - 1
        height_range = abs(end_leanto_height - start_leanto_height)
        
        # add roof for the stable
        for layer in range(height_range):
            geometry.placeCuboid(editor,
                (start_x, start_leanto_height - layer, start_z + layer),
                (end_x, start_leanto_height - layer, start_z + layer),
                Block("dark_oak_stairs[facing=north]"),
            )

        # add posts for the stable
        for post in range(self.width//2):
            spacing = post*2 
            geometry.placeCuboid(editor,
                (start_x + spacing, origin[1], start_z + layer),
                (start_x + spacing, origin[1] + height_range - 3, start_z+height_range-1),
                Block("stripped_dark_oak_log"),
            )
            # add fence between posts
            geometry.placeCuboid(editor,
                (start_x + spacing+1, origin[1], start_z+height_range-1),
                (start_x + spacing+1, origin[1], start_z+height_range-1),
                Block("oak_fence"),
            )
                
        # add fence around the stable
        geometry.placeCuboid(editor,
            (start_x, origin[1], start_z),
            (start_x, origin[1], start_z+height_range-2),
            Block("oak_fence"),
        )
        geometry.placeCuboid(editor,
            (end_x, origin[1], start_z),
            (end_x, origin[1], start_z+height_range-2),
            Block("oak_fence"),
        )
        
        # add hay bales around the stable
        geometry.placeCuboid(editor,
            (start_x + 1, origin[1], start_z),
            (start_x + 1, origin[1], start_z+height_range-2),
            Block("hay_block"),
        )
        
        # add horse in the stable
        geometry.placeCuboid(editor,
            (start_x + 3, origin[1], start_z+height_range-3),
            (start_x + 3, origin[1] + height_range//2, start_z+height_range-3),
            Block("brown_wool"),
        )
        geometry.placeCuboid(editor,
            (start_x + 2, origin[1] + height_range//2, start_z+height_range-3),
            (start_x + 2, origin[1] + height_range//2, start_z+height_range-3),
            Block("brown_wool"),
        )
        geometry.placeCuboid(editor,
            (start_x + 4, (origin[1] + height_range//2)-1, start_z+height_range-3),
            (start_x + 4, (origin[1] + height_range//2)-1, start_z+height_range-3),
            Block("brown_wool"),
        )
        geometry.placeCuboid(editor,
            (start_x + 5, origin[1], start_z+height_range-3),
            (start_x + 5, (origin[1] + height_range//2)-1, start_z+height_range-3),
            Block("brown_wool"),
        )
        
    
    
    # MODERN
    # ------
    def add_flat_roof(self, editor, origin):
        roof_height = origin[1] + self.height
        geometry.placeCuboid(editor,
            (origin[0], roof_height, origin[2]),
            (origin[0] + self.width-1, roof_height, origin[2] + self.depth-1),
            Block("smooth_stone_slab[type=top]"),
        )
        
        # add glass pane fence: 
        geometry.placeCuboid(editor,
            (origin[0], roof_height+1, origin[2]),
            (origin[0] + self.width-1, roof_height, origin[2]),
            Block("glass_pane"),
        )
        geometry.placeCuboid(editor,
            (origin[0] + self.width-1, roof_height+1, origin[2]),
            (origin[0] + self.width-1, roof_height, origin[2] + self.depth-1),
            Block("glass_pane"),
        )
        geometry.placeCuboid(editor,
            (origin[0], roof_height+1, origin[2] + self.depth-1),
            (origin[0] + self.width-1, roof_height, origin[2] + self.depth-1),
            Block("glass_pane"),
        )
        geometry.placeCuboid(editor,
            (origin[0], roof_height+1, origin[2] + self.depth-1),
            (origin[0], roof_height, origin[2]),
            Block("glass_pane"),
        )

    def add_concrete_walls(self, editor, origin):
        """concrete walls for modern houses"""
        self._place_hollow_cuboid(editor,
                                (0, 0, 0),
                                (self.width-1, self.height-1, self.depth-1),
                                "white_concrete",
                                origin,
                                thickness=1)

    def add_iron_door(self, editor, origin):
        """Place oak door in the pre-made opening."""
        ground_height = MinecraftUtils.get_ground_height(editor)

        door_x = origin[0] + self.width // 2 - 0.5
        door_z = origin[2]  
        
        # remove air for door opening
        geometry.placeCuboid(editor,
            (door_x, ground_height, door_z),
            (door_x + 1, ground_height, door_z),
            Block("air")
        )
        
        geometry.placeCuboid(editor,
            (door_x, ground_height, door_z),    
            (door_x, ground_height, door_z),    
            Block("iron_door[facing=south, half=lower, hinge=right]")
        )
        geometry.placeCuboid(editor,
            (door_x+1, ground_height, door_z),    
            (door_x+1, ground_height, door_z),    
            Block("iron_door[facing=south, half=lower, hinge=left]")
        )

    def add_floor_to_ceiling_windows(self, editor, origin):
        """floor to ceiling windows for modern houses"""
        ground_height = MinecraftUtils.get_ground_height(editor)
        window_y = ground_height + 1  
        
        left_window_x = origin[0] + 2  # 2 blocks from left edge
        right_window_x = origin[0] + self.width - 4  # 2 blocks from right edge
        
        # only place on front wall 
        dz = origin[2]  
        
        # remove air for left window opening
        geometry.placeCuboid(editor,
            (left_window_x, window_y, dz),
            (left_window_x + 1, window_y + self.height-3, dz),
            Block("air")
        )
        # remove air for right window opening
        geometry.placeCuboid(editor,
            (right_window_x, window_y, dz),
            (right_window_x + 1, window_y + self.height-3, dz),
            Block("air")
        )
        
        # add glass pane for left window
        geometry.placeCuboid(editor,
            (left_window_x, window_y, dz),
            (left_window_x + 1, window_y + self.height-3, dz),
            Block("glass_pane")
        )
        
        # add glass pane for right window
        geometry.placeCuboid(editor,
            (right_window_x, window_y, dz),
            (right_window_x + 1, window_y + self.height-3, dz),
            Block("glass_pane")
        )

    def add_gate(self, editor, origin):
        """Modern gate at property entrance."""
        gate_width = 4
        gate_x = origin[0] + self.width // 2 - gate_width//2
        
        # gate posts 
        post_height = origin[1] + self.height // 2
        geometry.placeCuboid(editor,
            (gate_x, origin[1], origin[2] - 3),
            (gate_x, post_height, origin[2] - 3),
            Block("smooth_quartz")
        )
        geometry.placeCuboid(editor,
            (gate_x + gate_width, origin[1], origin[2] - 3),
            (gate_x + gate_width, post_height, origin[2] - 3),
            Block("smooth_quartz")
        )
        
        # horizontal beam connecting posts
        geometry.placeCuboid(editor,
            (gate_x, post_height, origin[2] - 3),
            (gate_x + gate_width, post_height, origin[2] - 3),
            Block("smooth_quartz_slab[type=bottom]")
        )
        
        # sliding gate panels (using iron bars)
        panel_height = 3
        geometry.placeCuboid(editor,
            (gate_x + 1, origin[1], origin[2] - 3),
            (gate_x + gate_width - 1, origin[1] + panel_height-1, origin[2] - 3),
            Block("iron_bars")
        )

    def add_pool(self, editor, origin):
        pool_x_start = origin[0]
        pool_z_start = origin[2] + self.depth
        pool_width = self.width
        pool_depth = 6
        pool_height = origin[1]
        
        # add pool frame
        geometry.placeCuboid(editor,
            (pool_x_start, pool_height, pool_z_start),
            (pool_x_start + pool_width-1, pool_height, pool_z_start + pool_depth-1),
            Block("smooth_stone_slab[type=bottom]"),
        )
        geometry.placeCuboid(editor,
            (pool_x_start, pool_height + 1, pool_z_start),
            (pool_x_start + pool_width-1, pool_height + 1, pool_z_start),
            Block("smooth_stone_slab[type=bottom]"),
        )
        geometry.placeCuboid(editor,
            (pool_x_start + pool_width-1, pool_height + 1, pool_z_start),
            (pool_x_start + pool_width - 1, pool_height + 1, pool_z_start + pool_depth-1),
            Block("smooth_stone_slab[type=bottom]"),
        )
        geometry.placeCuboid(editor,
            (pool_x_start, pool_height + 1, pool_z_start + pool_depth-1),
            (pool_x_start + pool_width - 1, pool_height + 1, pool_z_start + pool_depth-1),
            Block("smooth_stone_slab[type=bottom]"),
        )
        geometry.placeCuboid(editor,
            (pool_x_start, pool_height + 1, pool_z_start + pool_depth-1),
            (pool_x_start, pool_height + 1, pool_z_start),
            Block("smooth_stone_slab[type=bottom]"),
        )
        
        # add glass fence outline
        geometry.placeCuboid(editor,
            (pool_x_start, pool_height + 1, pool_z_start),
            (pool_x_start + pool_width-1, pool_height + 1, pool_z_start),
            Block("glass_pane"),
        )
        geometry.placeCuboid(editor,
            (pool_x_start + pool_width-1, pool_height + 1, pool_z_start),
            (pool_x_start + pool_width - 1, pool_height + 1, pool_z_start + pool_depth-1),
            Block("glass_pane"),
        )
        geometry.placeCuboid(editor,
            (pool_x_start, pool_height + 1, pool_z_start + pool_depth-1),
            (pool_x_start + pool_width - 1, pool_height + 1, pool_z_start + pool_depth-1),
            Block("glass_pane"),
        )
        geometry.placeCuboid(editor,
            (pool_x_start, pool_height + 1, pool_z_start + pool_depth-1),
            (pool_x_start, pool_height + 1, pool_z_start),
            Block("glass_pane"),
        )
        
        # add water
        geometry.placeCuboid(editor,
            (pool_x_start+1, pool_height+1, pool_z_start+1),
            (pool_x_start + pool_width-2, pool_height+1, pool_z_start + pool_depth-2),
            Block("water"),
        )
    
    
    
    # HAUNTED
    # --------
    def add_broken_roof(self, editor, origin):
        """A-frame barn roof with glass pane center."""
        ground_height = MinecraftUtils.get_ground_height(editor)
        
        # calculate how many layers until left and right sides meet
        layers_needed = (self.width // 2) + 1
        roof_height = min(layers_needed, 8)
        
        if roof_height % 2 == 1:  
            center_layer = roof_height // 2
        else:  
            center_layer = (roof_height // 2) - 1
        
        # build the A-frame rows of stairs
        for layer in range(roof_height):
            y_level = ground_height + self.height + layer 
            
            
            left_x = origin[0] + layer - 1
            right_x = origin[0] + self.width - layer
            
            cobweb_depth = randint(1, self.depth-1)
            
            # place left side stairs
            geometry.placeCuboid(
                editor, 
                (left_x, y_level, origin[2] - 1), 
                (left_x, y_level, origin[2] + self.depth), 
                Block("crimson_stairs", {"facing": "east"})
            )
            
            geometry.placeCuboid(
                editor, 
                (left_x, y_level, origin[2] + cobweb_depth), 
                (left_x, y_level, origin[2] + cobweb_depth), 
                Block("cobweb")
            )
            
            # place right side stairs
            geometry.placeCuboid(
                editor,
                (right_x, y_level, origin[2] - 1),
                (right_x, y_level, origin[2] + self.depth),
                Block("crimson_stairs", {"facing": "west"})
            )
            geometry.placeCuboid(
                editor, 
                (right_x, y_level, origin[2] + cobweb_depth), 
                (right_x, y_level, origin[2] + cobweb_depth), 
                Block("cobweb")
            )
            
            # fill the space between stairs 
            if left_x + 1 < right_x:
                geometry.placeCuboid(
                    editor,
                    (left_x + 1, y_level, origin[2]),
                    (right_x - 1, y_level, origin[2] + self.depth - 1),
                    Block("dark_oak_planks")
                )
                
                if layer == center_layer:
                    center_x = (left_x + right_x) // 2
                    geometry.placeCuboid(editor,
                        (center_x, y_level, origin[2]+1),
                        (center_x, y_level, origin[2]+1),
                        Block("air")
                    )
                    geometry.placeCuboid(editor,
                        (center_x+1, y_level, origin[2]+1),
                        (center_x+1, y_level, origin[2]+1),
                        Block("air")
                    )
                    
                    # add torches behind window (eerie light on)
                    editor.placeBlock(
                        (center_x, y_level, origin[2]+1),
                        Block("torch")
                    )
                    editor.placeBlock(
                        (center_x+1, y_level, origin[2]+1),
                        Block("torch")
                    )
                    
                    
                    editor.placeBlock(
                        (center_x, y_level, origin[2]),
                        Block("glass_pane")
                    )
                    editor.placeBlock(
                        (center_x+1, y_level, origin[2]),
                        Block("glass_pane")
                    )

    def add_dark_wood_walls(self, editor, origin):
        """dark oak planks walls for haunted houses"""
        self._place_hollow_cuboid(editor,
                                (0, 0, 0),
                                (self.width-1, self.height-1, self.depth-1),
                                "dark_oak_planks",
                                origin,
                                thickness=1)

    def add_damaged_door(self, editor, origin):
        """double spruce doors for haunted houses"""
        ground_height = MinecraftUtils.get_ground_height(editor)
        
        door_x = origin[0] + self.width // 2 - 0.5
        door_z = origin[2]  
        
        # remove air for door opening
        geometry.placeCuboid(editor,
            (door_x, ground_height, door_z),
            (door_x + 1, ground_height, door_z),
            Block("air")
        )
        
        geometry.placeCuboid(editor,
            (door_x, ground_height, door_z),
            (door_x, ground_height, door_z),   
            Block("spruce_door[facing=south, half=lower, hinge=right]")
        )
        geometry.placeCuboid(editor,
            (door_x+1, ground_height, door_z),
            (door_x+1, ground_height, door_z),
            Block("spruce_door[facing=south, half=lower, hinge=left]")
        )

    def add_destroyed_windows(self, editor, origin):
        """Destroyed windows with cobwebs for haunted houses"""
        ground_height = MinecraftUtils.get_ground_height(editor)
        window_y = ground_height + (self.height // 2)  
        
        left_window_x = origin[0] + 2  # 2 blocks from left edge
        right_window_x = origin[0] + self.width - 4  # 2 blocks from right edge
        
        # only place on front wall 
        dz = origin[2]  
        
        # remove air for left window opening
        geometry.placeCuboid(editor,
            (left_window_x, window_y, dz),
            (left_window_x + 1, window_y + 1, dz),
            Block("air")
        )
        # remove air for right window opening
        geometry.placeCuboid(editor,
            (right_window_x, window_y, dz),
            (right_window_x + 1, window_y + 1, dz),
            Block("air")
        )
        
        # add cobweb for left window
        geometry.placeCuboid(editor,
            (left_window_x, window_y, dz),
            (left_window_x, window_y, dz),
            Block("cobweb")
        )
        geometry.placeCuboid(editor,
            (left_window_x+1, window_y+1, dz),
            (left_window_x+1, window_y+1, dz),
            Block("cobweb")
        )
        
        # add cobweb for right window
        geometry.placeCuboid(editor,
            (right_window_x, window_y, dz),
            (right_window_x, window_y, dz),
            Block("cobweb")
        )
        geometry.placeCuboid(editor,
            (right_window_x+1, window_y+1, dz),
            (right_window_x+1, window_y+1, dz),
            Block("cobweb")
        )

    def add_creepy_entrance(self, editor, origin):
        entrance_x_start = origin[0] + self.width // 2 - 2
        entrance_z_start = origin[2] - 6
        entrance_width = 4
        
        for i in range(3):
            spacing = i * 2
            # add right posts with jack o lanterns
            geometry.placeCuboid(editor,
                (entrance_x_start, origin[1], entrance_z_start + spacing),
                (entrance_x_start, origin[1] + 1, entrance_z_start + spacing),
                Block("spruce_fence")
            )
            geometry.placeCuboid(editor,
                (entrance_x_start, origin[1] + 2, entrance_z_start + spacing),
                (entrance_x_start, origin[1] + 2, entrance_z_start + spacing),
                Block("jack_o_lantern")
            )
            
            # add left posts with jack o lanterns
            geometry.placeCuboid(editor,
                (entrance_x_start + entrance_width-1, origin[1], entrance_z_start + spacing),
                (entrance_x_start + entrance_width-1, origin[1] + 1, entrance_z_start + spacing),
                Block("spruce_fence")
            )
            geometry.placeCuboid(editor,
                (entrance_x_start + entrance_width-1, origin[1] + 2, entrance_z_start + spacing),
                (entrance_x_start + entrance_width-1, origin[1] + 2, entrance_z_start + spacing),
                Block("jack_o_lantern")
            )

    def add_grave_yard(self, editor, origin):
        grave_yard_x_start = origin[0]
        grave_yard_z_start = origin[2] + self.depth + 2
        grave_width = 2
        grave_depth = 3
        
        # add posts for the stable
        for row in range(self.width//4):
            spacing = row*4 
            
            # add dirt mound
            geometry.placeCuboid(editor,
                (grave_yard_x_start + grave_width + spacing, origin[1], grave_yard_z_start),
                (grave_yard_x_start + spacing, origin[1], grave_yard_z_start + grave_depth),
                Block("dirt"),
            )
            
            # add grave stone
            geometry.placeCuboid(editor,
                (grave_yard_x_start + grave_width + spacing, origin[1], grave_yard_z_start-1),
                (grave_yard_x_start + grave_width + spacing - 2, origin[1]+2, grave_yard_z_start-1),
                Block("chiseled_stone_bricks"),
            )
            
            # add zombie head
            geometry.placeCuboid(editor,
                (grave_yard_x_start + grave_width + spacing-1, origin[1]+1, grave_yard_z_start),
                (grave_yard_x_start + grave_width + spacing-1, origin[1]+1, grave_yard_z_start),
                Block("zombie_head[rotation=8]"),
            )
            
        
    
    
    # HELPERS FOR REWARD FUNCTION
    # ------------------------------------------------------------
    def has_overlap(self):
        roof_count = sum(1 for prop in self.props if "roof" in prop)
        wall_count = sum(1 for prop in self.props if "wall" in prop)
        door_count = sum(1 for prop in self.props if "door" in prop)
        window_count = sum(1 for prop in self.props if "window" in prop)
        deco1_count = sum(1 for prop in self.props if "deco1" in prop)
        deco2_count = sum(1 for prop in self.props if "deco2" in prop)
        return roof_count > 1 or wall_count > 1 or door_count > 1 or window_count > 1 or deco1_count > 1 or deco2_count > 1
    
    def has_roof(self):
        return self.FARM_ROOF in self.props or self.MEDIEVAL_ROOF in self.props or self.MODERN_ROOF in self.props or self.HAUNTED_ROOF in self.props
    def has_wall(self):
        return self.FARM_WALL in self.props or self.MEDIEVAL_WALL in self.props or self.MODERN_WALL in self.props or self.HAUNTED_WALL in self.props
    def has_door(self):
        return self.FARM_DOOR in self.props or self.MEDIEVAL_DOOR in self.props or self.MODERN_DOOR in self.props or self.HAUNTED_DOOR in self.props
    def has_window(self):
        return self.FARM_WINDOW in self.props or self.MEDIEVAL_WINDOW in self.props or self.MODERN_WINDOW in self.props or self.HAUNTED_WINDOW in self.props
    def has_deco1(self):
        return self.FARM_PORCH in self.props or self.MEDIEVAL_CHIMNEY in self.props or self.MODERN_GATE in self.props or self.HAUNTED_ENTRANCE in self.props
    def has_deco2(self):
        return self.FARM_CROP in self.props or self.MEDIEVAL_STABLE in self.props or self.MODERN_POOL in self.props or self.HAUNTED_GRAVE_YARD in self.props
    
    def is_farmhouse(self):
        return self.FARM_ROOF in self.props and self.FARM_WALL in self.props and self.FARM_DOOR in self.props and self.FARM_WINDOW in self.props and self.FARM_PORCH in self.props and self.FARM_CROP in self.props
    def is_medieval(self):
        return self.MEDIEVAL_ROOF in self.props and self.MEDIEVAL_WALL in self.props and self.MEDIEVAL_DOOR in self.props and self.MEDIEVAL_WINDOW in self.props and self.MEDIEVAL_CHIMNEY in self.props and self.MEDIEVAL_STABLE in self.props
    def is_modern(self):
        return self.MODERN_ROOF in self.props and self.MODERN_WALL in self.props and self.MODERN_DOOR in self.props and self.MODERN_WINDOW in self.props and self.MODERN_GATE in self.props and self.MODERN_POOL in self.props
    def is_haunted(self):
        return self.HAUNTED_ROOF in self.props and self.HAUNTED_WALL in self.props and self.HAUNTED_DOOR in self.props and self.HAUNTED_WINDOW in self.props and self.HAUNTED_ENTRANCE in self.props and self.HAUNTED_GRAVE_YARD in self.props
    
    
    # HELPERS FOR PLACING MINECRAFT WALLS
    # ------------------------------------------------------------

    def _place_cuboid(self, editor, start_relative, end_relative, block, absolute_origin):
        """Place a cuboid relative to the house origin."""
        x0, y0, z0 = absolute_origin
        sx, sy, sz = start_relative
        ex, ey, ez = end_relative
        
        # Place using GDPC's geometry.placeCuboid
        geometry.placeCuboid(editor, 
                            (x0 + sx, y0 + sy, z0 + sz),
                            (x0 + ex, y0 + ey, z0 + ez),
                            Block(block))

    def _place_hollow_cuboid(self, editor, start_relative, end_relative, block, absolute_origin, thickness=1):
        """Place a hollow cuboid (walls only)."""
        x0, y0, z0 = absolute_origin
        sx, sy, sz = start_relative
        ex, ey, ez = end_relative
        
        # Outer shell
        self._place_cuboid(editor, start_relative, end_relative, block, absolute_origin)
        
        # Hollow out interior (subtract thickness from all sides)
        if (ex - sx) > 2*thickness and (ez - sz) > 2*thickness:
            inner_start = (sx + thickness, sy, sz + thickness)
            inner_end = (ex - thickness, ey, ez - thickness)
            # Place air inside
            self._place_cuboid(editor, inner_start, inner_end, "air", absolute_origin)