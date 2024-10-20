import cairo
import cv2
import math
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
from PIL import Image
from pyrr import Matrix44, Vector3
from typing import Callable

NEAR = 0.1
FAR = 10.0
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.0001)
STEREO_CALIB_FLAGS = cv2.CALIB_FIX_INTRINSIC  + cv2.CALIB_ZERO_DISPARITY

def run():
    # read simulation parameters, and replace missing values with defaults
    sim_data = pandas.read_csv("./assets/sims.csv")
    with open("./run/sim_parameters.tex", "w") as file:
        file.write(sim_data.to_latex(index=False, na_rep=""))
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
    with open("./run/sim_positions.tex", "w") as file:
        file.write(target_placements.to_latex(index=False, na_rep=""))
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
    #
    # # Distortion Shader setup
    # with open("assets/shaders/distortion-vert.glsl", "r") as file:
    #     dis_vert = file.read();
    # with open("assets/shaders/distortion-frag.glsl", "r") as file:
    #     dis_frag = file.read();
    # distortion_pass = ctx.program (
    #     vertex_shader= dis_vert,
    #     fragment_shader= dis_frag,
    # )

    # Load target model as vertex buffer obj
    pattern_vbo = ctx.buffer(parse_obj_file("./assets/models/plane.obj").flatten())
    pattern_vao = ctx.vertex_array(pattern_pass, [(pattern_vbo, "3f 3f 2f", "in_position", "in_normal", "in_texcoord")]) 
    cam_up = Vector3([0.0, 1.0, 0.0])
    cam_front = Vector3([0.0, 0.0, 1.0])
    # screen_quad = numpy.array([
    #     [-1.0, -1.0, 1.0, 0.0],
    #     [-1.0,  1.0, 1.0, 1.0],
    #     [ 1.0,  1.0, 0.0, 1.0],
    #
    #     [ 1.0,  1.0, 0.0, 1.0],
    #     [ 1.0, -1.0, 0.0, 0.0],
    #     [-1.0, -1.0, 1.0, 0.0],
    # ], dtype='f4')
    # screen_vbo = ctx.buffer(screen_quad.flatten())
    # # screen_vao = ctx.vertex_array(distortion_pass, [(screen_vbo, "2f 2f", "in_position", "in_texcoord")])
    # screen_vao = ctx.vertex_array(distortion_pass, [])
    # screen_vao.vertices = 3


    simdir = "./run/sims"
    if os.path.exists(simdir):
        for sim in os.listdir(simdir):
            targ = os.path.join(simdir, sim)
            if os.path.isdir(targ):
                shutil.rmtree(targ)
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
        keys = [
            "Horizontal Resolution",
            "Vertical Resolution",
            "Texture Resolution",
            "Camera Distance",
            "Grid Size",
            "Horizontal FOV",
            "Valid Positions",
            "Used Positions",
            "Radial Distortion k1",
            "Radial Distortion k2",
            "Tangential Distortion p1",
            "Tangential Distortion p2",
            "Mean Camera X Error",
            "Mean Camera Y Error",
            "Mean Camera Z Error",
            "Mean Camera Pitch Error",
            "Mean Camera Yaw Error",
            "Mean Camera Roll Error",
            "Mean Positional Error",
            "Mean Rotational Error",
            "Mean Reprojection Error Left",
            "Mean Reprojection Error Right",
        ];
        results = []
        results.append({ key: [] for key in keys })
        results.append({ key: [] for key in keys })
        Path(f"./run/sims/{name}/captures").mkdir(exist_ok=True, parents=True)
        Path(f"./run/sims/{name}/plots").mkdir(exist_ok=True)
        while iterator.isRunning():
            firstLast = iterator.isFirstOrLast()

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

            l_img_pts = [[], []]
            r_img_pts = [[], []]
            obj_pts = [[], []]
            successful = [0, 0]
            # r_imgs = [[], []]
            # l_imgs = [[], []]

            cam_offset = float(camDist()) / 2
            
            for texture in [1, 0]:
                found = 0
                if texture:
                    grid_size = (squares - 1, squares - 1)
                else:
                    grid_size = (squares , int(squares / 2 ))
                objp = numpy.zeros(((grid_size[0])*(grid_size[1]),3), numpy.float32)
                objp[:,:2] = numpy.mgrid[0:grid_size[0],0:grid_size[1]].T.reshape(-1,2)
                for posID, position in enumerate(positions):
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

                    image = cv2.undistort(numpy.array(image), camera_matrix, dist_coeffs)
                    cv2.imwrite("./run/cvl.jpg", image)
                    l_cv_img = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGBA2GRAY)

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

                    image = cv2.undistort(numpy.array(image), camera_matrix, dist_coeffs)
                    cv2.imwrite("./run/cvr.jpg", image)

                    r_cv_img = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGBA2GRAY)
                    if texture:
                        r_ret, r_geom = cv2.findChessboardCorners(r_cv_img, grid_size, None, flags=cv2.CALIB_CB_ADAPTIVE_THRESH)
                        l_ret, l_geom = cv2.findChessboardCorners(l_cv_img, grid_size, None, flags=cv2.CALIB_CB_ADAPTIVE_THRESH)
                    else:
                        r_ret, r_geom = cv2.findCirclesGrid(r_cv_img, (int(squares/2), squares), None, flags=cv2.CALIB_CB_SYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING)
                        l_ret, l_geom = cv2.findCirclesGrid(l_cv_img, (int(squares/2), squares), None, flags=cv2.CALIB_CB_SYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING)


                    # Save the image captures for use in the report
                    if firstLast:
                        l_hl = cv2.drawChessboardCorners(l_cv_img, grid_size,l_geom, l_ret)
                        r_hl = cv2.drawChessboardCorners(r_cv_img, grid_size,r_geom, r_ret)

                        file = f"./run/sims/{name}/captures/"
                        if firstLast == 1:
                            file += "first"
                        else:
                            file += "last"
                        if texture:
                            file += "-square"
                        else:
                            file += "-circle"
                        file += f"-{posID}"
                        cv2.imwrite(f"{file}-left.jpg", l_hl)
                        cv2.imwrite(f"{file}-right.jpg", r_hl)
                        stereo_bm = cv2.StereoBM_create(numDisparities=16, blockSize=15)
                        disp_map = stereo_bm.compute(l_cv_img, r_cv_img)
                        disp_map = cv2.normalize(disp_map, None, 0, 355, cv2.NORM_MINMAX)
                        cv2.imwrite(f"{file}-disparity.jpg", disp_map)

                    if l_ret & r_ret:
                        
                        found+=1
                        successful[texture] += 1

                        obj_pts[texture].append(objp)
                        # Refine corner points
                        l_geom = cv2.cornerSubPix(l_cv_img, l_geom , (11, 11), (-1, -1), CRITERIA)
                        r_geom = cv2.cornerSubPix(r_cv_img, r_geom , (11, 11), (-1, -1), CRITERIA)

                        l_img_pts[texture].append(l_geom)
                        r_img_pts[texture].append(r_geom)
                        # r_imgs[texture].append(r_cv_img)
                        # l_imgs[texture].append(l_cv_img)
                    else:
                        # Append nulls to keep arrays lengths at parity
                        obj_pts[texture].append([])
                        l_img_pts[texture].append([])
                        r_img_pts[texture].append([])
                        # r_imgs[texture].append([])
                        # l_imgs[texture].append([])

                    
                
            # Remove entries where any pattern failed to produce a valid calibration parameter match
            for index in range(len(obj_pts[0]) - 1, - 1, - 1):
               if not len(obj_pts[0][index]) or not len(obj_pts[1][index]):
                   del obj_pts[0][index]
                   del r_img_pts[0][index]
                   del l_img_pts[0][index]
                   del obj_pts[1][index]
                   del r_img_pts[1][index]
                   del l_img_pts[1][index]
            if len(obj_pts[0]) > 0:
               # ALL results must be recorded here to ensure that all the dicts lengths are equal 

                for texture in [0, 1]:
                    results[texture]["Horizontal Resolution"].append(h_pix)
                    results[texture]["Vertical Resolution"].append(v_pix)
                    results[texture]["Texture Resolution"].append(t_pix)
                    results[texture]["Camera Distance"].append(float(camDist()))
                    results[texture]["Grid Size"].append(squares)
                    results[texture]["Horizontal FOV"].append(int(fov()))
                    results[texture]["Radial Distortion k1"].append(float(r1()))
                    results[texture]["Radial Distortion k2"].append(float(r2()))
                    results[texture]["Tangential Distortion p1"].append(float(t1()))
                    results[texture]["Tangential Distortion p2"].append(float(t2()))
                    results[texture]["Used Positions"].append(len(obj_pts[0]))
                    results[texture]["Valid Positions"].append(successful[texture])
        
                    ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(obj_pts[texture], l_img_pts[texture], (h_pix, v_pix), None, None)
                    ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(obj_pts[texture], r_img_pts[texture], (h_pix, v_pix), None, None)
                    # Perform stereo calibration
                    ret, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(obj_pts[texture], l_img_pts[texture], r_img_pts[texture], mtx_left, dist_left, mtx_right, dist_right, (h_pix, v_pix),criteria=CRITERIA, flags=STEREO_CALIB_FLAGS)
                    # Construct the intrinsic matrices using the fov and aspect ratio
                    # fov_x = math.radians(float(fov()))
                    # fov_y = 2 * math.atan(math.tan(fov_x / 2) * (h_pix / v_pix))
                    # f_x = 1 / math.tan(fov_x/ 2)
                    # f_y = 1 / math.tan(fov_y/ 2)
                    #
                    # intrinsic_l = numpy.array([
                    #     [ f_x, 0, -cam_offset ],
                    #     [ 0, f_y, 0],
                    #     [ 0, 0, 1]
                    # ])
                    # intrinsic_r = numpy.array([
                    #     [ f_x, 0, cam_offset],
                    #     [ 0, f_y, 0],
                    #     [ 0, 0, 1]
                    # ])

                    # Create the projection matrices 
                    # proj_l = numpy.hstack((intrinsic_l, numpy.zeros((3, 1))))
                    # proj_r = numpy.hstack((intrinsic_r @ R, T.reshape(3, 1)))

                    posErr = [[], [], []]
                    rotErr = [[], [], []]
                    totPos = []
                    totRot = []

                    for _ in positions:
                        x = abs(T[0] - float(camDist()))
                        y = abs(T[1])
                        z = abs(T[2])
                        posErr[0].append(x)
                        posErr[1].append(y)
                        posErr[2].append(z)
                        totPos.append(x + y + z)

                        sy = numpy.sqrt(R[0,0] ** 2 + R[1, 0] ** 2)
                        if sy < 1e-6: # Gimbal Lock Prevention
                            roll = abs(numpy.arctan2(R[2, 1], R[2, 2]))
                            pitch = abs(numpy.arctan2(-R[2, 0], sy))
                            yaw = abs(numpy.arctan2(R[1, 0], R[0, 0]))
                        else:
                            roll = abs(numpy.arctan2(-R[1, 2], R[1, 1]))
                            pitch = abs(numpy.arctan2(-R[2, 0], sy))
                            yaw = 0
                        totRot.append(roll + pitch + yaw)
                        rotErr[1].append(roll)
                        rotErr[2].append(pitch)
                        rotErr[0].append(yaw)

                    results[texture]["Mean Camera X Error"].append(numpy.mean(posErr[0]))
                    results[texture]["Mean Camera Y Error"].append(numpy.mean(posErr[1]))
                    results[texture]["Mean Camera Z Error"].append(numpy.mean(posErr[2]))
                    results[texture]["Mean Positional Error"].append(numpy.mean(totPos))
                    results[texture]["Mean Camera Pitch Error"].append(numpy.mean(rotErr[0]))
                    results[texture]["Mean Camera Roll Error"].append(numpy.mean(rotErr[1]))
                    results[texture]["Mean Camera Yaw Error"].append(numpy.mean(rotErr[2]))
                    results[texture]["Mean Rotational Error"].append(numpy.mean(totRot))
                    
                    reprojErr = []
                    for index in range(len(obj_pts[texture])):
                        pts, _ = cv2.projectPoints(obj_pts[texture][index], rvecs_right[index], tvecs_right[index], mtx_right, dist_right)
                        reprojErr.append(cv2.norm(r_img_pts[texture][index], pts, cv2.NORM_L2)/len(pts))
                    results[texture]["Mean Reprojection Error Right"].append(numpy.mean(reprojErr))
                    reprojErr = []
                    for index in range(len(obj_pts[texture])):
                        pts, _ = cv2.projectPoints(obj_pts[texture][index], rvecs_left[index], tvecs_left[index], mtx_left, dist_left)
                        reprojErr.append(cv2.norm(l_img_pts[texture][index], pts, cv2.NORM_L2)/len(pts))
                    results[texture]["Mean Reprojection Error Left"].append(numpy.mean(reprojErr))





            iterator.increment()
        pandas.DataFrame(results[0]).to_csv(f"./run/sims/{name}/square.csv", index=False)
        pandas.DataFrame(results[1]).to_csv(f"./run/sims/{name}/circle.csv", index=False)



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

    def currentIteration(self):
        return self.iteration

    def isFirstOrLast(self):
        if self.iteration == 0:
            return 1
        if self.iteration == self.maxIteration - 1:
            return 2

        return 0

run()
