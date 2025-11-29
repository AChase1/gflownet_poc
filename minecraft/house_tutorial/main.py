from random import choice, randint
from gdpc import Editor, Block
from gdpc.geometry import placeCuboid, placeRectOutline, placeCuboidHollow
from gdpc.vector_tools import ivec3

from ..minecraft_utils import MinecraftUtils

if __name__ == "__main__":
    try: 
        
        editor = Editor(buffering=True) 
        
        MinecraftUtils.setup_world(editor)

        build_area = editor.getBuildArea()
        
        # start from empty area
        MinecraftUtils.clear_build_area(editor, build_area)
          
        y = MinecraftUtils.get_ground_height(editor)
        x = build_area.offset.x
        z = build_area.offset.z
        
        # helper to see the build area in the world
        MinecraftUtils.set_build_area_outline(editor, build_area, y)        
        
        height = randint(3, 7)
        depth  = randint(3, 10)
        floorPalette = [
            Block("stone_bricks"),
            Block("cracked_stone_bricks"),
            Block("cobblestone"),
        ]

        # randomize wall material
        wallBlock = choice([
            Block("oak_planks"),
            Block("spruce_planks"),
            Block("white_terracotta"),
            Block("green_terracotta"),
        ])

        # build the main house structure
        placeCuboidHollow(editor, (x, y, z), (x+4, y+height, z+depth), wallBlock)
        placeCuboid(editor, (x, y, z), (x+4, y, z+depth), floorPalette)

        # build the roof
        for dx in range(1, 4):
            yy = y + height + 2 - dx

            # build row of stairs blocks
            leftBlock  = Block("oak_stairs", {"facing": "east"})
            rightBlock = Block("oak_stairs", {"facing": "west"})
            placeCuboid(editor, (x+2-dx, yy, z-1), (x+2-dx, yy, z+depth+1), leftBlock)
            placeCuboid(editor, (x+2+dx, yy, z-1), (x+2+dx, yy, z+depth+1), rightBlock)

        # build the top row of the roof
        yy = y + height + 1
        placeCuboid(editor, (x+2, yy, z-1), (x+2, yy, z+depth+1), Block("oak_planks"))
        
        # build a door
        doorBlock = Block("oak_door", {"facing": "north", "hinge": "left"})
        editor.placeBlock((x+2, y+1, z), doorBlock)

        # clear some space in front of the door
        placeCuboid(editor, (x+1, y+1, z-1), (x+3, y+3, z-1), Block("air"))
        
        print("Finished tutorial")
        
    except Exception as e: 
        print(f"Caught error while executing tutorial: {e}")