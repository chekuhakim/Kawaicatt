import streamlit as st
import replicate
import os

# Set page config
st.set_page_config(page_title="Kawaii Cat Image Generator", page_icon="🐱")

st.title("Kawaii Cat Image Generator")

# API Token input
api_token = st.text_input("Enter your Replicate API Token:", type="password")

# Only show the rest of the UI if an API token is provided
if api_token:
    os.environ["REPLICATE_API_TOKEN"] = api_token

    # User inputs
    prompt = st.text_input("Enter your prompt:", "Illustration of C2T cat and cow on white background")
    width = st.slider("Width:", min_value=256, max_value=1024, value=500, step=64)
    height = st.slider("Height:", min_value=256, max_value=1024, value=500, step=64)
    lora_scale = st.slider("LoRA Scale:", min_value=0.0, max_value=1.0, value=0.79, step=0.01)
    guidance_scale = st.slider("Guidance Scale:", min_value=1.0, max_value=20.0, value=3.5, step=0.1)
    num_inference_steps = st.slider("Number of Inference Steps:", min_value=1, max_value=50, value=28)

    if st.button("Generate Image"):
        with st.spinner("Generating image..."):
            try:
                output = replicate.run(
                    "chekuhakim/kawaiicat:2b65183c4bac425a42b14fc801833bace01f525df7f4a5970c77208a8a174780",
                    input={
                        "model": "dev",
                        "width": width,
                        "height": height,
                        "prompt": prompt,
                        "lora_scale": lora_scale,
                        "num_outputs": 1,
                        "aspect_ratio": "1:1",
                        "output_format": "webp",
                        "guidance_scale": guidance_scale,
                        "output_quality": 80,
                        "extra_lora_scale": 0.8,
                        "num_inference_steps": num_inference_steps
                    }
                )
                
                if output:
                    st.image(output[0], caption="Generated Image", use_column_width=True)
                else:
                    st.error("Failed to generate image. Please try again.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

else:
    st.warning("Please enter your Replicate API Token to use the generator.")

st.markdown("---")
st.markdown("Created with ❤️ using Replicate API and Streamlit")
