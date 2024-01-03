import subprocess

def download_cyclegan_model(model_name):
    script_content = f"""
    FILE={model_name}

    echo "Note: available models are summer2winter_yosemite, winter2summer_yosemite, style_monet, style_cezanne, style_ukiyoe, style_vangogh"

    echo "Specified [$FILE]"

    mkdir -p ./checkpoints/${{FILE}}_pretrained
    MODEL_FILE=./checkpoints/${{FILE}}_pretrained/iter_100_net_G.pth
    URL=http://efrosgans.eecs.berkeley.edu/cyclegan/pretrained_models/$FILE.pth

    wget -N $URL -O $MODEL_FILE
    """

    # Write the script content to a temporary file
    script_path = "download_cyclegan_model.sh"
    with open(script_path, "w") as script_file:
        script_file.write(script_content)

    # Run the script using subprocess
    subprocess.run(["bash", script_path])

    # Remove the temporary script file
    subprocess.run(["rm", script_path])


