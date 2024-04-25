import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types
from pprint import pprint
from fiftyone import ViewField as F

import fiftyone.core.utils as fou
import numpy as np
import fiftyone.zoo as foz
import fiftyone.core.storage as fos
import base64
import os
import subprocess
import sys
import torch
from PIL import Image, ImageDraw, ImageFont


class ImageTo3D(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="ImageTo3D",
            label="ImageTo3D",
            description="Converts an image to a 3D model with shap_e",
            icon="/assets/cube.svg",
            dynamic=True,

        )
    
    def resolve_input(self, ctx):
        inputs = types.Object()

        ready = _image_inputs(ctx, inputs)
        checked = check_shap_e(ctx, inputs)
        _execution_mode(ctx, inputs)
        


        return types.Property(inputs, view=types.View(label="Import samples"))

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)

    def execute(self, ctx):
        import_type = ctx.params.get("import_type", None)


        _image_to_3d(ctx)
         

class TextTo3D(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="TextTo3D",
            label="TextTo3D",
            description="Converts a text prompt to a 3D model with shap_e",
            icon="/assets/cube.svg",
            dynamic=True,

        )
    
    def resolve_input(self, ctx):
        inputs = types.Object()

        ready = _text_inputs(ctx, inputs)
        checked = check_shap_e(ctx, inputs)
        _execution_mode(ctx, inputs)
        


        return types.Property(inputs, view=types.View(label="Import samples"))

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)

    def execute(self, ctx):
        import_type = ctx.params.get("import_type", None)


        _text_to_3d(ctx)

def check_shap_e(ctx, inputs):
    if os.path.exists("shap_e"):
        inputs.view(
            "notice",
            types.Notice(
                label=(
                    "Do not forget to \"pip intall -e .\" in your ./shap_e directory before running " 
                )
            ),
        )
    else:
        models_dir = os.path.expanduser("~/fiftyone/__models__")
        shap_e_dir = os.path.join(models_dir, "shap_e")

        if not os.path.exists(shap_e_dir):
            # Clone the shap_e repository
            try:
                subprocess.run(["git", "clone", "https://github.com/openai/shap-e.git", shap_e_dir], check=True)
                print("shap_e cloned successfully.")
            except subprocess.CalledProcessError as e:
                print("Error:", e)
        else:
            inputs.view(
            "notice",
            types.Notice(
                label=(
                    "Do not forget to \"pip intall -e .\" in your ~/fiftyone/__models__/shap_e directory before running!" 
                )
            ),
        )

    return True

def _image_inputs(ctx, inputs):

    ready = False
    inputs.obj(
        "media_file",
        required=True,
        label="Media file",
        description="Choose a media file to add to this dataset",
        view=types.FileView(label="Media file"),
    )

    ready = bool(ctx.params.get("media_file", None))

    if not ready:
        return False

    inputs.list(
        "tags",
        types.String(),
        default=None,
        label="Tags",
        description="An optional list of tags to give each new sample",
    )

    ready = _upload_media_inputs(ctx, inputs)
    if not ready:
        return False

    return True

def _text_inputs(ctx, inputs):

    ready = False
    inputs.str(
            "prompt",
            label="Prompt",
            description=(
                "Provide the prompt to use to generate a 3D model"
            ),
        )



    inputs.list(
        "tags",
        types.String(),
        default=None,
        label="Tags",
        description="An optional list of tags to give each new sample",
    )

    file_explorer = types.FileExplorerView(
        choose_dir=True,
        button_label="Choose a directory...",
    )
    inputs.file(
        "upload_dir",
        required=True,
        label="Upload directory",
        description="Provide a directory into which to upload the media",
        view=file_explorer,
    )
    upload_dir = _parse_path(ctx, "upload_dir")

    if upload_dir is None:
        return False


    return True

def _upload_media_inputs(ctx, inputs):

    file_explorer = types.FileExplorerView(
        choose_dir=True,
        button_label="Choose a directory...",
    )
    inputs.file(
        "upload_dir",
        required=True,
        label="Upload directory",
        description="Provide a directory into which to upload the media",
        view=file_explorer,
    )
    upload_dir = _parse_path(ctx, "upload_dir")

    if upload_dir is None:
        return False

    inputs.bool(
        "overwrite",
        default=False,
        required=False,
        label="Overwrite existing",
        description=(
            "Do you wish to overwrite existing media of the same name "
            "(True) or append a unique suffix when necessary to avoid "
            "name clashses (False)"
        ),
        view=types.CheckboxView(),
    )

    return True


def _upload_media_bytes(ctx):
    media_obj = ctx.params["media_file"]
    upload_dir = _parse_path(ctx, "upload_dir")
    overwrite = ctx.params["overwrite"]
    filename = media_obj["name"]
    content = base64.b64decode(media_obj["content"])

    if overwrite:
        outpath = fos.join(upload_dir, filename)
    else:
        filename_maker = fou.UniqueFilenameMaker(output_dir=upload_dir)
        outpath = filename_maker.get_output_path(input_path=filename)

    fos.write_file(content, outpath)
    return outpath

def _text_to_3d(ctx):
    prompt = ctx.params.get("prompt", None)
    tags = ctx.params.get("tags", None)

    samples = []
    group = fo.Group()

    upload_dir = _parse_path(ctx, "upload_dir")

    image = Image.new("RGB", (640, 640), color="black")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("DejaVuSans.ttf", size=80)
    text_width, text_height = draw.textsize(prompt, font)
    position = ((640 - text_width) // 2, (640 - text_height) // 2)

    # Write text on the image
    draw.text(position, prompt, fill="white", font=font)



    filename_maker = fou.UniqueFilenameMaker(output_dir=upload_dir)
    filepath = filename_maker.get_output_path(output_ext=".png")
    print(upload_dir)
    print(filepath)
    image.save(filepath)

    sample = fo.Sample(filepath=filepath, tags=tags, group=group.element("Input"))
    samples.append(sample)
    if not os.path.exists("shap_e"):
        sys.path.append("~/fiftyone/__models__")

    from shap_e.diffusion.sample import sample_latents
    from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
    from shap_e.models.download import load_model, load_config
    from shap_e.util.notebooks import decode_latent_mesh



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    xm = load_model('transmitter', device=device)
    model = load_model('text300M', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))
    batch_size = 4
    guidance_scale = 15.0

    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(texts=[prompt] * batch_size),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )

    mesh_path = filepath.split(".")[0] + ".ply"
    scene_path = filepath.split(".")[0] + ".fo3d"

    for latent in latents:
        t = decode_latent_mesh(xm, latent).tri_mesh()
        with open(mesh_path, 'wb') as f:
            t.write_ply(f)


    ply = fo.PlyMesh(name="TextTo3D_Shap-e", ply_path=mesh_path)

    scene = fo.Scene(camera=fo.PerspectiveCamera(up="Z"))
    scene.add(ply)   
    scene.write(scene_path)
    
    sample = fo.Sample(scene_path, group=group.element("Shap-e Mesh"))
    samples.append(sample)
    ctx.dataset.add_samples(samples)

    return

def _image_to_3d(ctx):
    style = ctx.params.get("style", None)
    tags = ctx.params.get("tags", None)

    samples = []
    group = fo.Group()

    filepath = _upload_media_bytes(ctx)

    sample = fo.Sample(filepath=filepath, tags=tags, group=group.element("Input"))
    samples.append(sample)
    if not os.path.exists("shap_e"):
        sys.path.append("~/fiftyone/__models__")

    from shap_e.diffusion.sample import sample_latents
    from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
    from shap_e.models.download import load_model, load_config
    from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
    from shap_e.util.image_util import load_image
    from shap_e.util.notebooks import decode_latent_mesh


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    xm = load_model('transmitter', device=device)
    model = load_model('image300M', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))

    batch_size = 4
    guidance_scale = 3.0

    # To get the best result, you should remove the background and show only the object of interest to the model.
    image = load_image(filepath)

    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(images=[image] * batch_size),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )

    mesh_path = filepath.split(".")[0] + ".ply"
    scene_path = filepath.split(".")[0] + ".fo3d"

    for latent in latents:
        t = decode_latent_mesh(xm, latent).tri_mesh()
        with open(mesh_path, 'wb') as f:
            t.write_ply(f)


    ply = fo.PlyMesh(name="ImageTo3D_Shap-e", ply_path=mesh_path)

    scene = fo.Scene(camera=fo.PerspectiveCamera(up="Z"))
    scene.add(ply)   
    scene.write(scene_path)
    
    sample = fo.Sample(scene_path, group=group.element("Shap-e Mesh"))
    samples.append(sample)
    ctx.dataset.add_samples(samples)

    return





def _parse_path(ctx, key):
    value = ctx.params.get(key, None)
    return value.get("absolute_path", None) if value else None

def _execution_mode(ctx, inputs):
    delegate = ctx.params.get("delegate", False)

    if delegate:
        description = "Uncheck this box to execute the operation immediately"
    else:
        description = "Check this box to delegate execution of this task"

    inputs.bool(
        "delegate",
        default=False,
        label="Delegate execution?",
        description=description,
        view=types.CheckboxView(),
    )

    if delegate:
        inputs.view(
            "notice",
            types.Notice(
                label=(
                    "You've chosen delegated execution. Note that you must "
                    "have a delegated operation service running in order for "
                    "this task to be processed. See "
                    "https://docs.voxel51.com/plugins/using_plugins.html#delegated-operations "
                    "for more information"
                )
            ),
        )

def register(plugin):
    plugin.register(ImageTo3D)
    plugin.register(TextTo3D)
