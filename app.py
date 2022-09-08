import gradio as gr
import os
from utils.default_models import ensure_default_models
import sys
import traceback
from pathlib import Path
from time import perf_counter as timer
import numpy as np
import torch
from encoder import inference as encoder
from synthesizer.inference import Synthesizer
#from toolbox.utterance import Utterance
from vocoder import inference as vocoder
import time
import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf
import argparse
from utils.argutils import print_args

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

# Maximum of generated wavs to keep on memory
MAX_WAVS = 15
utterances = set()
current_generated = (None, None, None, None) # speaker_name, spec, breaks, wav
synthesizer = None # type: Synthesizer
current_wav = None
waves_list = []
waves_count = 0
waves_namelist = []

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
#encoder.load_model(args.enc_model_fpath)
#synthesizer = Synthesizer(args.syn_model_fpath)
#vocoder.load_model(args.voc_model_fpath)

def compute_embedding(in_fpath):

    if not encoder.is_loaded():
        model_fpath = args.enc_model_fpath
        print("Loading the encoder %s... " % model_fpath)
        start = time.time() 
        encoder.load_model(model_fpath)
        print("Done (%dms)." % int(1000 * (time.time() - start)), "append")


    ## Computing the embedding
    # First, we load the wav using the function that the speaker encoder provides. This is
    
    # Get the wav from the disk. We take the wav with the vocoder/synthesizer format for
    # playback, so as to have a fair comparison with the generated audio
    wav = Synthesizer.load_preprocess_wav(in_fpath)
    
    # important: there is preprocessing that must be applied.

    # The following two methods are equivalent:
    # - Directly load from the filepath:
    preprocessed_wav = encoder.preprocess_wav(wav)

    # - If the wav is already loaded:
    #original_wav, sampling_rate = librosa.load(str(in_fpath))
    #preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)

    # Compute the embedding
    embed, partial_embeds, _ = encoder.embed_utterance(preprocessed_wav, return_partials=True)


    print("Loaded file succesfully")

    # Then we derive the embedding. There are many functions and parameters that the
    # speaker encoder interfaces. These are mostly for in-depth research. You will typically
    # only use this function (with its default parameters):
    #embed = encoder.embed_utterance(preprocessed_wav)
    
    return embed 
def create_spectrogram(text,embed):
        # If seed is specified, reset torch seed and force synthesizer reload
        if args.seed is not None:
            torch.manual_seed(args.seed)
            synthesizer = Synthesizer(args.syn_model_fpath)
        
        
        # Synthesize the spectrogram
        model_fpath = args.syn_model_fpath
        print("Loading the synthesizer %s... " % model_fpath)
        start = time.time()
        synthesizer = Synthesizer(model_fpath)
        print("Done (%dms)." % int(1000 * (time.time()- start)), "append")          
        

        # The synthesizer works in batch, so you need to put your data in a list or numpy array
        texts = [text]
        embeds = [embed]
        # If you know what the attention layer alignments are, you can retrieve them here by
        # passing return_alignments=True
        specs = synthesizer.synthesize_spectrograms(texts, embeds)
        breaks = [spec.shape[1] for spec in specs]
        spec = np.concatenate(specs, axis=1)
        sample_rate=synthesizer.sample_rate
        return spec, breaks , sample_rate


def generate_waveform(current_generated):

        speaker_name, spec, breaks = current_generated
        assert spec is not None

        ## Generating the waveform
        print("Synthesizing the waveform:")
        # If seed is specified, reset torch seed and reload vocoder
        if args.seed is not None:
            torch.manual_seed(args.seed)
            vocoder.load_model(args.voc_model_fpath)

        model_fpath = args.voc_model_fpath
        # Synthesize the waveform
        if not vocoder.is_loaded():
            print("Loading the vocoder %s... " % model_fpath)
            start = time.time()
            vocoder.load_model(model_fpath)
            print("Done (%dms)." % int(1000 * (time.time()- start)), "append")    

        current_vocoder_fpath= model_fpath
        def vocoder_progress(i, seq_len, b_size, gen_rate):
            real_time_factor = (gen_rate / Synthesizer.sample_rate) * 1000
            line = "Waveform generation: %d/%d (batch size: %d, rate: %.1fkHz - %.2fx real time)" \
                % (i * b_size, seq_len * b_size, b_size, gen_rate, real_time_factor)
            print(line, "overwrite")       


        # Synthesizing the waveform is fairly straightforward. Remember that the longer the
        # spectrogram, the more time-efficient the vocoder.
        if  current_vocoder_fpath is not None:
            print("")
            generated_wav = vocoder.infer_waveform(spec, progress_callback=vocoder_progress)
        else:
            print("Waveform generation with Griffin-Lim... ")
            generated_wav = Synthesizer.griffin_lim(spec)

        print(" Done!", "append")


        ## Post-generation
        # There's a bug with sounddevice that makes the audio cut one second earlier, so we
        # pad it.
        generated_wav = np.pad(generated_wav, (0, Synthesizer.sample_rate), mode="constant")

        # Add breaks
        b_ends = np.cumsum(np.array(breaks) * Synthesizer.hparams.hop_size)
        b_starts = np.concatenate(([0], b_ends[:-1]))
        wavs = [generated_wav[start:end] for start, end, in zip(b_starts, b_ends)]
        breaks = [np.zeros(int(0.15 * Synthesizer.sample_rate))] * len(breaks)
        generated_wav = np.concatenate([i for w, b in zip(wavs, breaks) for i in (w, b)])


        # Trim excess silences to compensate for gaps in spectrograms (issue #53)
        generated_wav = encoder.preprocess_wav(generated_wav)


        return generated_wav


def save_on_disk(generated_wav,sample_rate):
        # Save it on the disk
        filename = "cloned_voice.wav"
        print(generated_wav.dtype)
        #OUT=os.environ['OUT_PATH']
        # Returns `None` if key doesn't exist
        #OUT=os.environ.get('OUT_PATH')
        #result = os.path.join(OUT, filename)
        result = filename
        print(" > Saving output to {}".format(result))
        sf.write(result, generated_wav.astype(np.float32), sample_rate)
        print("\nSaved output as %s\n\n" % result) 
      
        return  result     
def play_audio(generated_wav,sample_rate):
        # Play the audio (non-blocking)
        if not args.no_sound:
          
            try:
                sd.stop()
                sd.play(generated_wav, sample_rate)
            except sd.PortAudioError as e:
                print("\nCaught exception: %s" % repr(e))
                print("Continuing without audio playback. Suppress this message with the \"--no_sound\" flag.\n")
            except:
                raise
         

def clean_memory():
    import gc
    #import GPUtil
    # To see memory usage
    print('Before clean ')
    #GPUtil.showUtilization()
    #cleaning memory 1
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)
    print('After Clean GPU')
    #GPUtil.showUtilization()

def clone_voice(in_fpath, text):
    try:       
            speaker_name = "output"
            # Compute embedding
            embed=compute_embedding(in_fpath)
            print("Created the embedding")
            # Generating the spectrogram
            spec, breaks, sample_rate = create_spectrogram(text,embed)
            current_generated = (speaker_name, spec, breaks)
            print("Created the mel spectrogram")

            # Create waveform
            generated_wav=generate_waveform(current_generated)
            print("Created the the waveform ")

            # Save it on the disk
            save_on_disk(generated_wav,sample_rate)

            #Play the audio 
            #play_audio(generated_wav,sample_rate)

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
def greet(Text,Voicetoclone ,input_mic=None):
    text= "%s" % (Text)
    #reference_files= "%s" % (Voicetoclone)

    clean_memory()
    print(text,len(text),type(text))
    print(Voicetoclone,type(Voicetoclone))

    if  len(text) == 0 : 
        print("Please add text to the program")
        Text="Please add text to the program, thank you."
        is_no_text=True
    else:
        is_no_text=False

    
    if Voicetoclone==None and input_mic==None:
        print("There is no input audio")
        Text="Please add audio input, to the program, thank you."
        Voicetoclone='trump.mp3'
        if  is_no_text:
            Text="Please add text and audio, to the program, thank you."

    if  input_mic != None:
        # Get the wav file from the microphone
        print('The value of MIC IS :',input_mic,type(input_mic))
        Voicetoclone= input_mic


    text= "%s" % (Text)
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
      in_fpath = Path(Voicetoclone)
      #in_fpath= in_fpath.replace("\"", "").replace("\'", "")
      
      out_path=clone_voice(in_fpath, text)

      print(" > text: {}".format(text))

      print("Generated Audio")
      return "cloned_voice.wav"

demo = gr.Interface(
    fn=greet, 
    inputs=[gr.inputs.Textbox(label='What would you like the voice to say? (max. 2000 characters per request)'),
            gr.Audio(
            type="filepath",         
            source="upload",
            label='Please upload a voice to clone (max. 30mb)'),
            gr.inputs.Audio(
            source="microphone", 
            label='or record',
            type="filepath", 
            optional=True)

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
                        ["I am the cloned version of Donald Trump. Well.  I think what's happening to this country is unbelievably bad. We're no longer a respected country" ,"trump.mp3",]
                                           
                      ]     

    
    
    
    
    
    )
demo.launch()