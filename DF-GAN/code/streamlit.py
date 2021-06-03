import streamlit as st
from inference import setup, single_sampling, get_data, build_dictionary

# st.title('Simple Streamlit App')
# st.text('HELLO')
# s = st.text_input('type a name in the box below')
# st.write(f'Hello {s}')

app_formal_name = "DF-GAN Storybook"

@st.cache
def one_time_setup():
    gan_args = setup()
    return gan_args

device = "cuda"

with st.beta_expander("Make your own storybook!"):
    sentence = st.text_area(
        "Input your image description here to generate an image to use for your storybook!"
    )

    text_encoder, netG = one_time_setup()

    st.info("Encoding your story...")
    tokenized_description = get_data(sentence)
    captions = build_dictionary(tokenized_description)
    st.write(f'{tokenized_description}')

    fake_imgs = single_sampling(text_encoder, netG, captions, device)
    ## fake_imgs = single_inference(encoded_description)

    st.markdown(f'## Your Storybook: \n')
    # images = fake_imgs.squeeze(0).transpose(0, 1)[i].squeeze(0)
    # images = images_to_numpy(images)
    # image = PIL.Image.fromarray(images)

    
    st.image(fake_imgs, use_column_width='always')
    st.text(sentence)