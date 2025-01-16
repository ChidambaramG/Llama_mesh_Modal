import modal
import os
from os import listdir
from os.path import isfile, join
import gradio as gr
import trimesh
import numpy as np
import tempfile
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
from trimesh.exchange.gltf import export_glb


# Define Modal stub and persistent volume
HF_TOKEN = os.environ.get("HF_TOKEN", None)
app = modal.App("llama-mesh-app")
volume = modal.Volume.from_name("llama-mesh-volume", create_if_missing=True)

# Define Modal image with dependencies
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "gradio",
    "torch",
    "transformers",
    "accelerate",
    "tensorboard",
    "trimesh",
    "numpy",
    "tensorflow"
)

# Constants
DESCRIPTION = '''
<div>
<h1 style="text-align: center;">LLaMA-Mesh</h1>
<div>
<a style="display:inline-block" href="https://research.nvidia.com/labs/toronto-ai/LLaMA-Mesh/"><img src='https://img.shields.io/badge/public_website-8A2BE2'></a>
<a style="display:inline-block; margin-left: .5em" href="https://github.com/nv-tlabs/LLaMA-Mesh"><img src='https://img.shields.io/github/stars/nv-tlabs/LLaMA-Mesh?style=social'/></a>
</div>
<p>LLaMA-Mesh: Unifying 3D Mesh Generation with Language Models. <a style="display:inline-block" href="https://research.nvidia.com/labs/toronto-ai/LLaMA-Mesh/">[Project Page]</a> <a style="display:inline-block" href="https://github.com/nv-tlabs/LLaMA-Mesh">[Code]</a></p>
<p> Notice: (1) The default token length is 4096. If you observe incomplete generated meshes, try to increase the maximum token length into 8192.</p>
<p>(2) We only support generating a single mesh per dialog round. To generate another mesh, click the "clear" button and start a new dialog.</p>
<p>(3) If the LLM refuses to generate a 3D mesh, try adding more explicit instructions to the prompt, such as "create a 3D model of a table <strong>in OBJ format</strong>." A more effective approach is to request the mesh generation at the start of the dialog.</p>
</div>
'''

LICENSE = """
<p/>
---
Built with Meta Llama 3.1 8B
"""

PLACEHOLDER = """
<div style="padding: 30px; text-align: center; display: flex; flex-direction: column; align-items: center;">
   <h1 style="font-size: 28px; margin-bottom: 2px; opacity: 0.55;">LLaMA-Mesh</h1>
   <p style="font-size: 18px; margin-bottom: 2px; opacity: 0.65;">Create 3D meshes by chatting.</p>
</div>
"""

css = """
h1 {
  text-align: center;
  display: block;
}

#duplicate-button {
  margin: auto;
  color: white;
  background: #1565c0;
  border-radius: 100vh;
}
"""

# @app.function(image=image, gpu="T4", timeout=600, volumes={"/data": volume})
# def apply_gradient_color(mesh_text):
#     temp_file = "/data/temp_mesh"
#     with open(temp_file + ".obj", "w") as f:
#         f.write(mesh_text)

#     mesh = trimesh.load_mesh(temp_file + ".obj", file_type='obj')
#     vertices = mesh.vertices
#     y_values = vertices[:, 1]
#     y_normalized = (y_values - y_values.min()) / (y_values.max() - y_values.min())

#     colors = np.zeros((len(vertices), 4))
#     colors[:, 0] = y_normalized
#     colors[:, 2] = 1 - y_normalized
#     colors[:, 3] = 1.0

#     mesh.visual.vertex_colors = colors

#     glb_path = temp_file + ".glb"
#     with open(glb_path, "wb") as f:
#         f.write(export_glb(mesh))
    
#     return glb_path

# @app.function(image=image, gpu="T4", timeout=600, volumes={"/data": volume})
# def visualize_mesh(mesh_text):
#     temp_file = "/data/temp_visualize_mesh.obj"
#     with open(temp_file, "w") as f:
#         f.write(mesh_text)
#     return temp_file

@app.function(image=image, gpu="A10G", timeout=6000, volumes={"/model_weights": volume})
@modal.web_endpoint(method="GET")
def run_gradio_app():
    def apply_gradient_color(mesh_text):
        temp_file = "/model_weights/data/temp_mesh"
        with open(temp_file + ".obj", "w") as f:
            f.write(mesh_text)

        mesh = trimesh.load_mesh(temp_file + ".obj", file_type='obj')
        vertices = mesh.vertices
        y_values = vertices[:, 1]
        y_normalized = (y_values - y_values.min()) / (y_values.max() - y_values.min())

        colors = np.zeros((len(vertices), 4))
        colors[:, 0] = y_normalized
        colors[:, 2] = 1 - y_normalized
        colors[:, 3] = 1.0

        mesh.visual.vertex_colors = colors

        glb_path = temp_file + ".glb"
        with open(glb_path, "wb") as f:
            f.write(export_glb(mesh))
        
        return glb_path
    
    def visualize_mesh(mesh_text):
        temp_file = "/model_weights/data/temp_visualize_mesh.obj"
        with open(temp_file, "w") as f:
            f.write(mesh_text)
        return temp_file

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    model_path = "/model_weights/Zhengyi/LLaMA-Mesh"
    # onlyfiles = [f for f in listdir(model_path) if isfile(join(model_path, f))]
    # print(onlyfiles)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    def chat_llama3_8b(message, history, temperature, max_new_tokens):
        conversation = []
        for user, assistant in history:
            conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
        conversation.append({"role": "user", "content": message})

        input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(model.device)
        
        streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)

        generate_kwargs = dict(
            input_ids=input_ids,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            eos_token_id=terminators,
        )
        if temperature == 0:
            generate_kwargs['do_sample'] = False
            
        t = Thread(target=model.generate, kwargs=generate_kwargs)
        t.start()

        outputs = []
        for text in streamer:
            outputs.append(text)
            yield "".join(outputs)
        # In the chat_llama3_8b function, replace the threaded code with:
        # output = model.generate(**generate_kwargs)

        # # Decode the output
        # decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

        # # Yield the decoded output
        # yield decoded_output

    chatbot = gr.Chatbot(height=450, placeholder=PLACEHOLDER, label='Gradio ChatInterface')

    with gr.Blocks(fill_height=True, css=css) as demo:
        with gr.Column(): 
            gr.Markdown(DESCRIPTION)
            with gr.Row():
                with gr.Column(scale=3):    
                    gr.ChatInterface(
                    fn=chat_llama3_8b,
                    chatbot=chatbot,
                    fill_height=True,
                    additional_inputs_accordion=gr.Accordion(label="⚙️ Parameters", open=False, render=False),
                    additional_inputs=[
                        gr.Slider(minimum=0,
                                maximum=1, 
                                step=0.1,
                                value=0.95, 
                                label="Temperature", 
                                render=False),
                        gr.Slider(minimum=128, 
                                maximum=8192,
                                step=1,
                                value=4096, 
                                label="Max new tokens", 
                                render=False),
                        ],
                    examples=[
                        ['Create a 3D model of a wooden hammer'],
                        ['Create a 3D model of a pyramid in obj format'],
                        ['Create a 3D model of a cabinet.'],
                        ['Create a low poly 3D model of a coffe cup'],
                        ['Create a 3D model of a table.'],
                        ["Create a low poly 3D model of a tree."],
                        ['Write a python code for sorting.'],
                        ['How to setup a human base on Mars? Give short answer.'],
                        ['Explain theory of relativity to me like I’m 8 years old.'],
                        ['What is 9,000 * 9,000?'],
                        ['Create a 3D model of a soda can.'],
                        ['Create a 3D model of a sword.'],
                        ['Create a 3D model of a wooden barrel'],
                        ['Create a 3D model of a chair.']
                        ],
                    cache_examples=False,
                                )
                gr.Markdown(LICENSE)
            
                with gr.Column(scale=2): 
                    output_model = gr.Model3D(label="3D Mesh Visualization", interactive=False)
                    gr.Markdown("You can copy the generated 3d objects in the left and paste in the textbox below. Put the button and you will see the visualization of the 3D mesh.")
                    
                    mesh_input = gr.Textbox(label="3D Mesh Input", placeholder="Paste your 3D mesh in OBJ format here...", lines=5)
                    visualize_button = gr.Button("Visualize 3D Mesh")
                    
                    visualize_button.click(fn=apply_gradient_color, inputs=[mesh_input], outputs=[output_model])
    
    return demo.launch(server_name="127.0.0.1", server_port=8000, share=True)
    # return demo

if __name__ == "__main__":
    app.serve()
    # interface = run_gradio_app()
    # interface.launch(server_name="127.0.0.1", server_port=8000, share=True)
