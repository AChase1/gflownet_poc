from gdpc import Block
from gdpc.geometry import placeRectOutline

# required commands for the Minecraft GDPC API
class MinecraftUtils:
    def __init__(self):
        pass

    @staticmethod
    def execute_command(editor, command):
      try: 
        editor.runCommand(command)
        print(f"Executed: {command}")
      except Exception as e:
        print(f"Command failed: {e}")
    
    # execute this command as soon as you enter the world
      # if you want to use player-relative coordinates, use '~' in front of each coordinate
    set_build_area = "setbuildarea 0 0 0 128 255 128"
    
    # use the following two commands to lock the time to noon (daylight all the time)
    set_time_noon = "time set noon"
    lock_daynight_cycle = "gamerule doDaylightCycle false"
    
    # this is used to clear large areas with the fill command, using air blocks
    # failing to do this will result in a "commandModificationBlockLimit exceeded" error
    modify_fill_limit = "gamerule commandModificationBlockLimit 10000000"
    
    # execute this command to clear the area with the fill command
    @staticmethod
    def clear_area_with_air(x1, y1, z1, x2, y2, z2):
      return f"fill {x1} {y1} {z1} {x2} {y2} {z2} air"
    
    # call this method at the beginning of every gdpc script, after fetching the editor instance
    @staticmethod
    def setup_world(editor):
      """Setup the world for the tutorial"""
      MinecraftUtils.execute_command(editor, MinecraftUtils.set_build_area)
      MinecraftUtils.execute_command(editor, MinecraftUtils.set_time_noon)
      MinecraftUtils.execute_command(editor, MinecraftUtils.lock_daynight_cycle)
      MinecraftUtils.execute_command(editor, MinecraftUtils.modify_fill_limit)
      
    @staticmethod
    def get_ground_height(editor):
        editor.loadWorldSlice(cache=True)
        heightmap = editor.worldSlice.heightmaps["MOTION_BLOCKING_NO_LEAVES"]
        ground_height = heightmap[3,3] - 1
        return ground_height
        
    # call this method to clear the build area by filling it with air blocks
    @staticmethod
    def clear_build_area(editor, build_area):
        print("Clearing area with fill command...")
        
        # expand the build area by 10 blocks in all directions (just in case extra blocks were created outside of it)
        x1 = build_area.offset.x-10
        z1 = build_area.offset.z-10
        x2 = x1 + build_area.size.x + 10
        z2 = z1 + build_area.size.z + 10
        y1 = -60  
        y2 = 200
        
        # replace all blocks in the build area with air
        MinecraftUtils.execute_command(editor, MinecraftUtils.clear_area_with_air(x1, y1, z1, x2, y2, z2))

    def set_build_area_outline(editor, build_area, y):
        placeRectOutline(editor, build_area.toRect(), y-1, Block("blue_concrete"))