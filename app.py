import gradio as gr
import os
import sys
import os
import string
import numpy as np
import IPython
from IPython.display import Audio
import torch
import argparse
import os
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf
import torch
from encoder import inference as encoder
from encoder.params_model import model_embedding_size as speaker_embedding_size
from synthesizer.inference import Synthesizer
from utils.argutils import print_args
from utils.default_models import ensure_default_models
from vocoder import inference as vocoder
import sounddevice as sd
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("-e", "--enc_model_fpath", type=Path,
                    default="saved_models/default/encoder.pt",
                    help="Path to a saved encoder")
parser.add_argument("-s", "--syn_model_fpath", type=Path,
                    default="saved_models/default/synthesizer.pt",
                    help="Path to a saved synthesizer")
parser.add_argument("-v", "--voc_model_fpath", type=Path,
                    default="saved_models/default/vocoder.pt",
                    help="Path to a saved vocoder")
parser.add_argument("--cpu", action="store_true", help=\
    "If True, processing is done on CPU, even when a GPU is available.")
parser.add_argument("--no_sound", action="store_true", help=\
    "If True, audio won't be played.")
parser.add_argument("--seed", type=int, default=None, help=\
    "Optional random number seed value to make toolbox deterministic.")
args = parser.parse_args()
arg_dict = vars(args)
print_args(args, parser)

# Hide GPUs from Pytorch to force CPU processing
if arg_dict.pop("cpu"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

print("Running a test of your configuration...\n")

if torch.cuda.is_available():
    device_id = torch.cuda.current_device()
    gpu_properties = torch.cuda.get_device_properties(device_id)
    ## Print some environment information (for debugging purposes)
    print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
        "%.1fGb total memory.\n" %
        (torch.cuda.device_count(),
        device_id,
        gpu_properties.name,
        gpu_properties.major,
        gpu_properties.minor,
        gpu_properties.total_memory / 1e9))
else:
    print("Using CPU for inference.\n")

## Load the models one by one.
print("Preparing the encoder, the synthesizer and the vocoder...")
ensure_default_models(Path("saved_models"))
encoder.load_model(args.enc_model_fpath)
synthesizer = Synthesizer(args.syn_model_fpath)
vocoder.load_model(args.voc_model_fpath)

def compute_embedding(in_fpath):
    ## Computing the embedding
    # First, we load the wav using the function that the speaker encoder provides. This is
    # important: there is preprocessing that must be applied.

    # The following two methods are equivalent:
    # - Directly load from the filepath:
    preprocessed_wav = encoder.preprocess_wav(in_fpath)
    # - If the wav is already loaded:
    original_wav, sampling_rate = librosa.load(str(in_fpath))
    preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
    print("Loaded file succesfully")

    # Then we derive the embedding. There are many functions and parameters that the
    # speaker encoder interfaces. These are mostly for in-depth research. You will typically
    # only use this function (with its default parameters):
    embed = encoder.embed_utterance(preprocessed_wav)
    
    return embed 
def create_spectrogram(text,embed, synthesizer ):
        # If seed is specified, reset torch seed and force synthesizer reload
        if args.seed is not None:
            torch.manual_seed(args.seed)
            synthesizer = Synthesizer(args.syn_model_fpath)
        # The synthesizer works in batch, so you need to put your data in a list or numpy array
        texts = [text]
        embeds = [embed]
        # If you know what the attention layer alignments are, you can retrieve them here by
        # passing return_alignments=True
        specs = synthesizer.synthesize_spectrograms(texts, embeds)
        spec = specs[0]
        return spec

def generate_waveform(spec):
        ## Generating the waveform
        print("Synthesizing the waveform:")
        # If seed is specified, reset torch seed and reload vocoder
        if args.seed is not None:
            torch.manual_seed(args.seed)
            vocoder.load_model(args.voc_model_fpath)
        # Synthesizing the waveform is fairly straightforward. Remember that the longer the
        # spectrogram, the more time-efficient the vocoder.
        generated_wav = vocoder.infer_waveform(spec)

        ## Post-generation
        # There's a bug with sounddevice that makes the audio cut one second earlier, so we
        # pad it.
        generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")

        # Trim excess silences to compensate for gaps in spectrograms (issue #53)
        generated_wav = encoder.preprocess_wav(generated_wav)
        return generated_wav


def save_on_disk(generated_wav,synthesizer):
        # Save it on the disk
        filename = "cloned_voice.wav"
        print(generated_wav.dtype)
        #OUT=os.environ['OUT_PATH']
        # Returns `None` if key doesn't exist
        #OUT=os.environ.get('OUT_PATH')
        #result = os.path.join(OUT, filename)
        result = filename
        print(" > Saving output to {}".format(result))
        sf.write(result, generated_wav.astype(np.float32), synthesizer.sample_rate)
        print("\nSaved output as %s\n\n" % result) 
      
        return  result     
def play_audio(generated_wav,synthesizer):
        # Play the audio (non-blocking)
        if not args.no_sound:
          
            try:
                sd.stop()
                sd.play(generated_wav, synthesizer.sample_rate)
            except sd.PortAudioError as e:
                print("\nCaught exception: %s" % repr(e))
                print("Continuing without audio playback. Suppress this message with the \"--no_sound\" flag.\n")
            except:
                raise

def clone_voice(in_fpath, text,synthesizer):
    try:       
            # Compute embedding
            embed=compute_embedding(in_fpath)
            print("Created the embedding")
            # Generating the spectrogram
            spec = create_spectrogram(text,embed,synthesizer)
            print("Created the mel spectrogram")

            # Create waveform
            generated_wav=generate_waveform(spec)
            print("Created the the waveform ")

            # Save it on the disk
            save_on_disk(generated_wav,synthesizer)

            #Play the audio 
            play_audio(generated_wav,synthesizer)

            return        
    except Exception as e:
        print("Caught exception: %s" % repr(e))
        print("Restarting\n")

# Set environment variables
home_dir = os.getcwd()
OUT_PATH=os.path.join(home_dir, "out/")
os.environ['OUT_PATH'] = OUT_PATH

# create output path
os.makedirs(OUT_PATH, exist_ok=True)

USE_CUDA = torch.cuda.is_available()  

os.system('pip install -q pydub ffmpeg-normalize')
CONFIG_SE_PATH = "config_se.json"
CHECKPOINT_SE_PATH = "SE_checkpoint.pth.tar"
def greet(Text,Voicetoclone):
    text= "%s" % (Text)
    #reference_files= "%s" % (Voicetoclone)
    reference_files= Voicetoclone
    print("path url")
    print(Voicetoclone)
    sample= str(Voicetoclone)
    os.environ['sample'] = sample
    size= len(reference_files)*sys.getsizeof(reference_files)
    size2= size / 1000000
    if (size2 > 0.012) or len(text)>2000:
      message="File is greater than 30mb or Text inserted is longer than 2000 characters. Please re-try with smaller sizes."
      print(message)
      raise SystemExit("File is greater than 30mb. Please re-try or Text inserted is longer than 2000 characters. Please re-try with smaller sizes.")
    else:

      env_var = 'sample'
      if env_var in os.environ:
            print(f'{env_var} value is {os.environ[env_var]}')
      else:
            print(f'{env_var} does not exist')
      #os.system(f'ffmpeg-normalize {os.environ[env_var]} -nt rms -t=-27 -o {os.environ[env_var]} -ar 16000 -f')
      in_fpath = Path(sample)
      #in_fpath= in_fpath.replace("\"", "").replace("\'", "")
      
      out_path=clone_voice(in_fpath, text,synthesizer)

      print(" > text: {}".format(text))

      print("Generated Audio")
      return "cloned_voice.wav"

demo = gr.Interface(
    fn=greet, 
    inputs=[gr.inputs.Textbox(label='What would you like the voice to say? (max. 2000 characters per request)'),
            gr.Audio(
            type="filepath",         
            source="upload",
            label='Please upload a voice to clone (max. 30mb)')
            ],
    outputs="audio",

    title = 'Clone Your Voice',
            description = 'A simple application that Clone Your Voice.  Wait one minute to process.',
            article = 
                        '''<div>
                            <p style="text-align: center"> All you need to do is record your voice, type what you want be say
                            ,then wait for compiling. After that click on Play/Pause for listen the audio. The audio is saved in an wav format.
                            For more information visit <a href="https://ruslanmv.com/">ruslanmv.com</a>
                            </p>
                        </div>''',

           examples = [
                        ["I am the cloned version of Donald Trump.Well, I am a Republican, and I would run as a Republican. And I have a lot of confidence in the Republican Party. I don't have a lot of confidence in the president. I think what's happening to this country is unbelievably bad. We're no longer a respected country.","trump.mp3"]
                    
                      ]     

    
    
    
    
    
    )
demo.launch()