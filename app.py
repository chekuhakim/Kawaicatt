import streamlit as st
import replicate
import os

# Set page config
st.set_page_config(page_title="Image Generator", page_icon="🖼️")

st.title("Image Generator")
st.markdown('<input type="text" name="username" style="display:none">', unsafe_allow_html=True)
# API Token input
api_token = st.text_input("Enter your Replicate API Token:", type="password")

# Only show the rest of the UI if an API token is provided
if api_token:
    os.environ["REPLICATE_API_TOKEN"] = api_token

    # Model selection
    model_options = {
        "Kawaii Cat": "chekuhakim/kawaiicat:2b65183c4bac425a42b14fc801833bace01f525df7f4a5970c77208a8a174780",
        "Face Tuning": "chekuhakim/facetuning:0c1969354ea19f77d820dff199f6b3717f165f1f7215157ad49645a00fd166b1",
        "Custom": "custom"
    }
    
    selected_model = st.selectbox("Select Model:", list(model_options.keys()))
    
    if selected_model == "Custom":
        custom_model = st.text_input("Enter custom model name:", 
                                     help="Format: username/model:version")

    # User inputs (keep all original inputs)
    prompt = st.text_area("Enter your prompt:", "Illustration of C2T cat and cow on white background")
    
    aspect_ratio = st.selectbox("Aspect Ratio:", 
                                ["1:1", "16:9", "21:9", "3:2", "2:3", "4:5", "5:4", "3:4", "4:3", "9:16", "9:21", "custom"])
    
    if aspect_ratio == "custom":
        col1, col2 = st.columns(2)
        with col1:
            width = st.number_input("Width:", min_value=256, max_value=1440, value=512, step=16)
        with col2:
            height = st.number_input("Height:", min_value=256, max_value=1440, value=512, step=16)
    
    num_outputs = st.slider("Number of outputs:", min_value=1, max_value=4, value=1)
    lora_scale = st.slider("LoRA Scale:", min_value=-1.0, max_value=2.0, value=1.0, step=0.1)
    num_inference_steps = st.slider("Number of Inference Steps:", min_value=1, max_value=50, value=28)
    
    model = st.selectbox("Model:", ["dev", "schnell"])
    
    guidance_scale = st.slider("Guidance Scale:", min_value=0.1, max_value=10.0, value=3.5, step=0.1)
    seed = st.number_input("Seed (optional):", value=-1, help="Set for reproducible generation. Leave as -1 for random seed.")
    
    extra_lora = st.text_input("Extra LoRA (optional):", 
                               help="e.g., 'fofr/flux-pixar-cars' or HuggingFace/CivitAI URLs")
    extra_lora_scale = st.slider("Extra LoRA Scale:", min_value=0.0, max_value=1.0, value=0.8, step=0.1)
    
    output_format = st.selectbox("Output Format:", ["webp", "jpg", "png"])
    output_quality = st.slider("Output Quality:", min_value=0, max_value=100, value=80)
    
    disable_safety_checker = st.checkbox("Disable Safety Checker")

    if st.button("Generate Image"):
        with st.spinner("Generating image..."):
            try:
                input_data = {
                    "prompt": prompt,
                    "aspect_ratio": aspect_ratio,
                    "num_outputs": num_outputs,
                    "lora_scale": lora_scale,
                    "num_inference_steps": num_inference_steps,
                    "model": model,
                    "guidance_scale": guidance_scale,
                    "output_format": output_format,
                    "output_quality": output_quality,
                    "disable_safety_checker": disable_safety_checker
                }
                
                if aspect_ratio == "custom":
                    input_data["width"] = width
                    input_data["height"] = height
                
                if seed != -1:
                    input_data["seed"] = seed
                
                if extra_lora:
                    input_data["extra_lora"] = extra_lora
                    input_data["extra_lora_scale"] = extra_lora_scale

                # Determine which model to use
                if selected_model == "Custom":
                    model_path = custom_model
                else:
                    model_path = model_options[selected_model]

                output = replicate.run(
                    model_path,
                    input=input_data
                )
                
                # Debug: Print the raw output
                st.write("Debug - Raw output:", output)

                if output and isinstance(output, list):
                    for i, img_url in enumerate(output):
                        st.image(img_url, caption=f"Generated Image {i+1}", use_column_width=True)
                elif output and isinstance(output, str):
                    st.image(output, caption="Generated Image", use_column_width=True)
                else:
                    st.error("Failed to generate image. Unexpected output format.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

else:
    st.warning("Please enter your Replicate API Token to use the generator.")

st.markdown("---")
st.markdown("Created with ❤️ using Replicate API and Streamlit")
