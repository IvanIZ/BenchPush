import pygame
import pymunk 
import pymunk.pygame_util
import math
import numpy as np
import os

class Renderer():

    def __init__(self, space, env_width, env_height, render_scale=35, background_color=(255, 255, 255), caption="Renderer", **kwargs):
        """
        :param space: Pymunk space to be rendered
        :param env_width: the width of the environment in Pymunk unit (in our case, meter)
        :param env_height: the height of the envionment in Pymunk unit (in our case, meter)
        :param render_scale: the scale for transformation from pymunk unit to pygame pixel
        """

        # get parameters
        self.goal_line = kwargs.get('goal_line', None)
        self.goal_point = kwargs.get('goal_point', None)
        self.goal_region = kwargs.get('goal_region', None)                  # a tuple (goal_center,region_size)
        self.clearance_boundary = kwargs.get('clearance_boundary', None)
        self.centered = kwargs.get('centered', False)

        # scale to convert from pymunk meter unit to pygame pixel unit
        self.render_scale = render_scale
        self.background_color = background_color

        # Initialize Pygame
        pygame.init()
        self.pygame_w, self.pygame_h = env_width * self.render_scale, env_height * self.render_scale
        self.window = pygame.display.set_mode((self.pygame_w, self.pygame_h))
        pygame.display.set_caption(caption)

        # set pygame y-axis same as pymunk y-axis
        pymunk.pygame_util.positive_y_is_up = True

        self.draw_options = pymunk.pygame_util.DrawOptions(self.window)

        # disable the draw collision point flag
        self.draw_options.flags &= ~pymunk.SpaceDebugDrawOptions.DRAW_COLLISION_POINTS

        # convert from pymunk meter unit to pygame pixel unit
        self.draw_options.transform = pymunk.Transform.scaling(self.render_scale)
        if self.centered:
            self.draw_options.transform = self.draw_options.transform.translated(env_width / 2, env_width / 2)

        self.space = space

        self.path = None

        self.teleop_paths = None
        self.teleop_path_thickness = 10


    def to_pygame(self, pymunk_point):
        """
        Convert Pymunk world coorniates (meter unit) to Pygame screen coordinates (pixel unit)
        """
        x = pymunk_point[0]
        y = pymunk_point[1]
        if self.centered:
            return int(x * self.render_scale + self.pygame_w / 2), int(self.pygame_h / 2 - y * self.render_scale)
        return int(x * self.render_scale), int(self.pygame_h - y * self.render_scale)

    
    def update_path(self, path):
        """
        Update the planned path for display for planning-based policies
        """
        self.path = path

    def add_teleop_paths(self, paths):
        self.teleop_paths = paths
    
    def reset(self, new_space):
        """
        Reset the renderer to display a new Pymunk space
        """
        self.space = new_space

    
    def display_planned_path(self):
        """
        Display the planned path given from a planner
        """
        pygame.draw.lines(
                self.window, (255, 0, 0), False,  # red color, not a closed shape
                [self.to_pygame(point) for point in self.path],  # Convert trajectory to Pygame coordinates
                1,  # Line thickness
            )

    
    def display_teleop_paths(self):
        """
        Display the planned path given from a planner
        """


        surface = self.window.convert_alpha()
        surface.fill((0, 0, 0, 0))

        pygame.draw.lines(
                surface, (255, 100, 100), False,  # red color, not a closed shape
                [self.to_pygame(point) for point in self.teleop_paths[0]],  # Convert trajectory to Pygame coordinates
                self.teleop_path_thickness,  # Line thickness
            )

        pygame.draw.lines(
                surface, (255, 222, 33), False,  # red color, not a closed shape
                [self.to_pygame(point) for point in self.teleop_paths[1]],  # Convert trajectory to Pygame coordinates
                self.teleop_path_thickness,  # Line thickness
            )

        pygame.draw.lines(
                surface, (0, 0, 255), False,  # red color, not a closed shape
                [self.to_pygame(point) for point in self.teleop_paths[2]],  # Convert trajectory to Pygame coordinates
                self.teleop_path_thickness,  # Line thickness
            )

        self.window.blit(surface, (0, 0))

    
    def display_goal_line(self):
        """
        Display the goal line for ship ice navigation
        """
        goal_line = self.to_pygame([self.goal_line, self.goal_line])[1]
        pygame.draw.line(
            self.window,
            (255, 255, 255),  # Line color (white)
            (0, goal_line),                # Start point (x1, y1)
            (self.pygame_w, goal_line),    # End point (x2, y2)
            6               # Line width
        )

    

    def display_goal_region(self):
        """
        Display goal regions for navigation tasks
        """
        """
        Display goal point for navigation tasks
        """
        pygame.draw.circle(
            self.window,
            (144, 238, 144),  # Circle color (green)
            self.to_pygame(self.goal_region[0]),  # Circle center
            self.goal_region[1] * self.render_scale,  # Circle radius
            0   # Circle thickness
        )


    def display_goal_point(self):
        """
        Display goal point for navigation tasks
        """
        pygame.draw.circle(
            self.window,
            (255, 255, 255),  # Circle color (white)
            self.to_pygame(self.goal_point),  # Circle center
            5,  # Circle radius
            0   # Circle thickness
        )
    
    def display_clearance_boundary(self):
        """
        Display clearance boundary for object pushing tasks
        """
        pygame.draw.polygon(
            self.window,
            # '#20FE20',  # Line color (green)
            (144, 238, 144),
            [self.to_pygame(point) for point in self.clearance_boundary],  # Convert boundary to Pygame coordinates
            3  # Line width
        )

    def render(self, save=False, path=None, manual_draw=False):
        self.window.fill(self.background_color)
        if not manual_draw:
            self.space.debug_draw(self.draw_options)
        
        else:
            static_list = ['wall', 'receptacle', 'corner', 'divider', 'column']
            static_shapes = [shape for shape in self.space.shapes if shape.label in static_list]
            dynamic_shapes = [shape for shape in self.space.shapes if shape.label not in static_list]
            for shape in static_shapes:
                pygame.draw.polygon(
                    self.window,
                    shape.color,
                    [self.to_pygame(shape.body.local_to_world((x, y))) for x, y in shape.get_vertices()]
                )

            for shape in dynamic_shapes:
                pygame.draw.polygon(
                    self.window,
                    shape.color,
                    [self.to_pygame(shape.body.local_to_world((x, y))) for x, y in shape.get_vertices()]
                )

        if self.path is not None:
            self.display_planned_path()

        if self.teleop_paths is not None:
            self.display_teleop_paths()
        
        if self.goal_line is not None:
            self.display_goal_line()

        if self.goal_region is not None:
            self.display_goal_region()
        
        if self.clearance_boundary is not None:
            self.display_clearance_boundary()

        ### could add goal region display here
        ###

        ### Goal point display
        if self.goal_point is not None:
            self.display_goal_point()

        pygame.display.update()

        if save:
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)  # Create directories if they don't exist

            pygame.image.save(self.window, path)


    
    def close(self):
        pygame.quit()
