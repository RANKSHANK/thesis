import cairo
import cv2
import moderngl
import numpy
import os
import pandas
import shutil
import sys
from generate_patterns import (
    chessboard,
    circleboard,
)
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from pyrr import Matrix44, Vector3
from typing import Callable

NEAR = 0.1
FAR = 10.0
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.0001)
STEREO_CALIB_FLAGS = cv2.CALIB_FIX_INTRINSIC  + cv2.CALIB_ZERO_DISPARITY

def run():
    # read simulation parameters, and replace missing values with defaults
    sim_data = pandas.read_csv("./assets/sims.csv")
    names = sim_data["Name"].to_numpy(na_value="MISSING NAME")
    iterations = sim_data["Iterations"].to_numpy(na_value=10)
    h_res = sim_data["Horizontal Resolution"].to_numpy(na_value=1920)
    v_res = sim_data["Vertical Resolution"].to_numpy(na_value=1080)
    t_res = sim_data["Texture Resolution"].to_numpy(na_value=4096)
    grids = sim_data["Grid Size"].to_numpy(na_value=14)

    dists = sim_data["Camera Offset"].to_numpy(na_value=0.16)
    fovs = sim_data["FOV Horizontal"].to_numpy(na_value=100.0)
    k1s = sim_data["Radial Distortion k1"].to_numpy(na_value=0)
    k2s = sim_data["Radial Distortion k2"].to_numpy(na_value=0)
    p1s = sim_data["Tangential Distortion p1"].to_numpy(na_value=0)
    p2s = sim_data["Tangential Distortion p2"].to_numpy(na_value=0)
    
    target_placements = pandas.read_csv("./assets/positions.csv")
    target_placements.fillna(0, inplace=True)
    positions = []
    for _, row in target_placements.iterrows():
        trans = Matrix44.from_translation((row["x"].astype(float), row["y"].astype(float), row["z"].astype(float)), "f4")
        rot = Matrix44.from_eulers((row["roll"].astype(float), row["pitch"].astype(float), row["yaw"].astype(float)), "f4")
        positions.append(trans @ rot)

    # OpenGL Context
    ctx = moderngl.create_context(require=430, standalone=True)
    # Scene Shader setup
    with open("assets/shaders/pattern-vert.glsl", "r") as file:
        pat_vert = file.read();
    with open("assets/shaders/pattern-frag.glsl", "r") as file:
        pat_frag = file.read();
    pattern_pass = ctx.program (
        vertex_shader= pat_vert,
        fragment_shader= pat_frag,
    )
    # Load target model as vertex buffer obj
    pattern_vbo = ctx.buffer(parse_obj_file("./assets/models/plane.obj").flatten())
    pattern_vao = ctx.vertex_array(pattern_pass, [(pattern_vbo, "3f 3f 2f", "in_position", "in_normal", "in_texcoord")]) 
    cam_up = Vector3([0.0, 1.0, 0.0])
    cam_front = Vector3([0.0, 0.0, 1.0])

    simdir = "./run/sims"
    if os.path.exists(simdir):
        for sim in os.listdir(simdir):
            targ = os.path.join(simdir, sim)
            # if os.path.isdir(targ):
            #     shutil.rmtree(targ)
    for index, name in enumerate(names):
        # Allows commenting out simulations for testing
        if name.startswith("#"):
            continue
        iterator = SimIterator(int(iterations[index]))
        hRes = parse_simulation_parameter(iterator, h_res[index])
        vRes = parse_simulation_parameter(iterator, v_res[index])
        tRes = parse_simulation_parameter(iterator, t_res[index])
        camDist = parse_simulation_parameter(iterator, dists[index])
        grid = parse_simulation_parameter(iterator, grids[index])
        fov = parse_simulation_parameter(iterator, fovs[index])
        r1 = parse_simulation_parameter(iterator, k1s[index])
        r2 = parse_simulation_parameter(iterator, k2s[index])
        t1 = parse_simulation_parameter(iterator, p1s[index])
        t2 = parse_simulation_parameter(iterator, p2s[index])
        frame_path = f"./run/sims/{name}"
        Path(frame_path).mkdir(exist_ok=True, parents=True)

        frame = 0
        frames = []
        while iterator.isRunning():

            squares = int(grid())
            h_pix = int(hRes())
            v_pix = int(vRes())
            t_pix = int(tRes())
            
            fbo = ctx.simple_framebuffer((h_pix, v_pix))
            # Construct a cairo surface to draw the cal patterns to
            cairo_sfc = cairo.ImageSurface(cairo.FORMAT_ARGB32, t_pix, t_pix)# pyright: ignore

            # Draw the chessboard cal pattern to the cairo surface
            cairo_ctx = cairo.Context(cairo_sfc)
            cells = int(grid())
            chessboard(cairo_ctx, cells_per_row=cells, pixels_per_row=t_pix)
            # Create a GL texture with the cairo surface
            chessboard_texture = ctx.texture((t_pix, t_pix), 4, data=cairo_sfc.get_data())
            # cairo is bgra, convert to RGBA for openGL
            chessboard_texture.swizzle = "BGRA"

            # Draw the circleboard cal pattern to the cairo surface
            cairo_ctx = cairo.Context(cairo_sfc)
            circleboard(cairo_ctx, cells_per_row=cells, pixels_per_row=t_pix)
            # Create a GL texture with the cairo surface
            circboard_texture = ctx.texture((t_pix, t_pix), 4, data=cairo_sfc.get_data())
            circboard_texture.swizzle = "BGRA"
            cairo_sfc.finish()

            dist_coeffs = numpy.array([float(r1()), float(r2()), float(t1()), float(t2()), 0])

            camera_matrix = numpy.array([
                [h_pix, 0, h_pix/2],
                [0, v_pix, v_pix/2],
                [0, 0, 1]
            ])

            proj = Matrix44.perspective_projection(
                float(fov()),
                h_pix/v_pix,
                NEAR,
                FAR
            )

            r_imgs = []
            l_imgs = []

            cam_offset = float(camDist()) / 2
            
            for texture in [1, 0]:
                found = 0
                if texture:
                    grid_size = (squares - 1, squares - 1)
                else:
                    grid_size = (squares , int(squares / 2 ))
                for posID, position in enumerate([positions[0]]):
                    cam_pos = (-cam_offset, 0, 0)
                    look_at = Matrix44.look_at(
                        cam_pos,
                        (cam_pos + cam_front),
                        cam_up,
                    )
                    fbo.use()
                    ctx.clear((100/255), 149/255, 237/255, 0.0)
                    if texture:
                        chessboard_texture.use()
                    else:
                        circboard_texture.use()
                    ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)# pyright: ignore
                    pattern_pass["Light"] = (0, 0, -1)
                    pattern_pass["MVP"].write((proj * look_at * position).astype("f4")) #pyright:ignore
                    pattern_vao.render(mode=moderngl.TRIANGLES)#pyright: ignore
                    image = Image.frombytes("RGBA", fbo.size, fbo.read(components=4), "raw", "RGBA", 0, -1)

                    l_imgs.append(Image.fromarray(cv2.undistort(numpy.array(image), camera_matrix, dist_coeffs)))

                    cam_pos = (cam_offset, 0, 0)
                    look_at = Matrix44.look_at(
                        cam_pos,
                        (cam_pos + cam_front),
                        cam_up,
                    )

                    fbo.use()
                    ctx.clear((100/255), 149/255, 237/255, 0.0)

                    ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)# pyright: ignore
                    pattern_pass["Light"] = (0, 0, -1)
                    pattern_pass["MVP"].write((proj * look_at * position).astype("f4")) #pyright:ignore
                    if texture:
                        chessboard_texture.use()
                    else:
                        circboard_texture.use()
                    ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)# pyright: ignore
                    pattern_vao.render(mode=moderngl.TRIANGLES)#pyright: ignore

                    image = Image.frombytes("RGBA", fbo.size, fbo.read(components=4), "raw", "RGBA", 0, -1)

                    r_imgs.append(Image.fromarray(cv2.undistort(numpy.array(image), camera_matrix, dist_coeffs)))

            width = 1920
            height = 1080*2
            half_h = int(height / 2)
            stitched = Image.new("RGB", (width, height))
            stitched.paste(r_imgs[0], (0,0, width, half_h))
            stitched.paste(r_imgs[1], (0, half_h,width, height))
            draw = ImageDraw.Draw(stitched)
            draw.line([(1, 0), (1, height)], fill="white", width = 2)
            draw.line([(width-1, 0), (width-1, height)], fill="white", width = 2)
            draw.line([(0, half_h), (width, half_h)], fill="white", width = 4)

            del(l_imgs)
            del(r_imgs)
            frame += 1
            frames.append(stitched)
            iterator.increment()
        width = 1920 * len(frames)
        height = 1080 * 2
        stitched = Image.new("RGB", (width, height))
        x_start = 0
        for frame in frames:
            stitched.paste(frame, (x_start, 0, x_start + 1920, height))
            x_start += 1920
        stitched.save(f"{frame_path}/all_iters.png")



def parse_simulation_parameter(iterator, sim_param) -> Callable:
    if type(sim_param) == str:
        nums = []
        for val in sim_param.split(':'):
            val = val.strip()
            try:
                num = int(val)
            except ValueError:
                try:
                    num = float(val)
                except:
                    print(f"Malformed range value, must be numeric, found {val}", file=sys.stderr)
                    sys.exit(1)
            nums.append(num)
        if len(nums) == 2:
            nums.append(iterator.getDefaultIterations())
        if len(nums) == 3:
            prog = iterator.create_iterable(nums[2])
            step = (nums[1] - nums[0]) / (nums[2] - 1)
            return lambda: nums[0] + step * prog()
        print(f"Error, invalid sim param string, found {sim_param}", file=sys.stderr)
        sys.exit(1)

    isType = lambda type: hasattr(sim_param, "dtype") and numpy.issubdtype(sim_param.dtype, type)
    # Constant values return themselves
    if isType(numpy.number) or isType(numpy.bool_):
        return lambda: sim_param
    if type(sim_param) == float or type(sim_param) == int:
        return lambda: sim_param

    print(f"Error, unknown type fed from simulation parameters, found {sim_param}", file=sys.stderr)
    sys.exit(1)

# Converts a waveform .obj to VBO format
def parse_obj_file(obj_file_path):
    vertices = []
    normals = []
    texcoords = []
    faces = []

    with open(obj_file_path, 'r') as file:
        for line in file:
            # Vertex = 'v x y z'
            if line.startswith('v '):
                vertex = [float(x) for x in line.strip().split()[1:]]
                vertices.append(vertex)
            # Normal = 'vn x y z'
            elif line.startswith('vn '):
                normal = [float(x) for x in line.strip().split()[1:]]
                normals.append(normal)
            # UV = 'vt u v w' W
            elif line.startswith('vt '):
                texcoord = [float(x) for x in line.strip().split()[1:3]]
                texcoords.append(texcoord)
            # Face = 'f vert_idx uv_idx norm_idx'
            elif line.startswith('f '):
                face = []
                face_data = line.strip().split()[1:]
                for data in face_data:
                    vertex_index, texcoord_index, normal_index = [int(x) if x else 0 for x in data.split('/')]
                    face.append((vertex_index - 1, normal_index - 1, texcoord_index - 1))
                faces.append(face)

    vertices_data = []

    for face in faces:
        for vertex_index, normal_index, texcoord_index in face:
            px, py, pz = vertices[vertex_index]
            nx, ny, nz = normals[normal_index]
            u, v = texcoords[texcoord_index]
            vertex_data = [px, py, pz, nx, ny, nz, u, v]
            vertices_data.append(vertex_data)

    return numpy.array(vertices_data).astype("f4")

class SimIterator:

    def __init__(self, num_iterations):
        self.current_iteration = 0
        self.num_iterations = num_iterations
        self.iteration = 0
        self.iterators = []
        self.iterations = []
        self.maxIteration = 1

    def create_iterable(self, iterations):
        leng = len(self.iterators)
        self.iterators.append(0)
        self.iterations.append(iterations)
        self.maxIteration *= iterations
        return lambda: self.iterators[leng]

    def increment(self):
        for index in range(0, len(self.iterators)):
            val = self.iterators[index] + 1
            if val < self.iterations[index]:
                self.iterators[index] = val
                break
            if index == len(self.iterators):
                return
            self.iterators[index] = 0
        self.iteration += 1

    def isRunning(self):
        return self.iteration < self.maxIteration

    def getDefaultIterations(self):
        return self.num_iterations

    def currentIteration(self) -> int:
        return self.iteration

    def isFirstOrLast(self):
        if self.iteration == 0:
            return 1
        if self.iteration == self.maxIteration - 1:
            return 2

        return 0

run()
