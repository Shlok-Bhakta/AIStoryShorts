import gc
import json
import os
import random
import subprocess
import time
import sys
from threading import Thread
from time import sleep
import openai
import os
import music21
import soundfile
import torch
from diffusers import DiffusionPipeline
from espnet2.bin.tts_inference import Text2Speech
from midi2audio import FluidSynth
from moviepy.audio.fx.volumex import volumex
from moviepy.editor import *
from moviepy.video.fx.all import crop, scroll
from music21 import converter, midi
from nltk import tokenize
from profanity_filter import ProfanityFilter
from samplings import temperature_sampling, top_p_sampling
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from custom_selenium import *

from custom_stable_pipe import StableDiffusionWalkPipeline

if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0), '\n')
else:
    print('No GPU available, using the CPU instead.\n')
    device = torch.device("cpu")


try:
    file = open("data/sessiondata.txt", "r")
except: 
    # TODO replace later with an auto generated config so The error doesnt exist
    print("Session Data File Not Found, Please check that ther is one under the data folder")
    print("if not please create a file called sessiondata.txt and just write the number 0 in it")
    print("make sure to press save on the file after typing 0")

sessionNum = int(file.read())
file.close()

file = open("data/sessiondata.txt", "w")
file.write(str(sessionNum+1))
file.close()


#sets the value of how many prompts to generate (they will all eventually turn into videos)
amountOfPrompts = 5
storageLocation = "videos"

#sets character limit per line
lineCharacterLimit = 20

# Schedule Times:
scheduleTimes = ["12:00AM", "6:00AM", "12:00PM", "5:00PM", "10:00PM"]

#models
gptModel = "KoboldAI/OPT-350M-Nerys-v2"
ttsModel = "espnet/kan-bayashi_ljspeech_joint_finetune_conformer_fastspeech2_hifigan"
#imageModel = "prompthero/openjourney-v2"
imageModel = "CompVis/stable-diffusion-v1-4"
musicModel = 'sander-wood/text-to-music'

useOnline = True

knownGoodMusic = "GoodMusic.wav"

openai.api_key = os.environ['OPENAIKEY']
print("SECTION 1 COMPLETE")
print("================================================================")
###############################################################################################################################



try:
    os.mkdir(f"{storageLocation}/videoSession{sessionNum}")
    for i in range(amountOfPrompts):
        os.mkdir(f"{storageLocation}/videoSession{sessionNum}/video{i+1}")
        os.mkdir(f"{storageLocation}/videoSession{sessionNum}/video{i+1}/images")
        os.mkdir(f"{storageLocation}/videoSession{sessionNum}/video{i+1}/speech")
except:
    print("files already made, skipping...")
    

print("SECTION 2 COMPLETE")
print("================================================================")
###############################################################################################################################



# Add more templates along with filepaths here [LOCATION 1]
mapping = {"[genre]": "themes/genres.txt",
           "[food]": "themes/foods.txt",
           "[emotion]": "themes/emotions.txt",
           "[location]": "themes/locations.txt",
           "[object]": "themes/objects.txt",
           "[name]": "themes/names.txt",
           "[animal]": "themes/animals.txt",
           "[color]": "themes/colors.txt",
           }

# Add more generation templates here [LOCATION 2]
Templates = ["A [genre] story with a [object].", 
             "An adventure in [location] with a [emotion] [object].", 
             "A [emotion] story in [location].", 
             "A [genre] story in [location].",
             "A [emotion] [object] once",
             "A [emotion] [object] eating [food].",
             "A [object] once felt [emotion]",
             "[food] tastes good with [object]",
             "When life gives you [food] you make [object]",
             "When you are [emotion] eat some [food]",
             "When you are a [genre] you make a [object]",
             "[name] won the best [object]",
             "A [animal] named [name] once",
             "The [animal] was [emotion]",
             "A story about a [animal] in a [location]",
             "A [genre] story with [name]",
             "A [animal] went to a [location]",
             "the [color] [food]",
             ]


def replaceTemplateWithWord(prompt):
    """This function replaces all the [someing] with actial words from a file. The [] and the path is at LOCATION 1

    Args:
        prompt (_type_): A string that needs [] replaced with words

    Returns:
        String _type_: A string that has all the [] replaced with words from LOCATION 1
    """    
    output = prompt
    for key, value in mapping.items():
        if key in output:
            try:
                file = open(value, "r")
            except:
                print(f"check LOCATION 1 to make sure that the file path is correct for {key} (check for typos 😉)")
            data = file.read().split("\n")
            data = data[random.randint(0, len(data) - 1)]
            output = output.replace(key, data)
    
    return output
    

# Generate the prompts for the video
prompts = []
for i in range(amountOfPrompts):
    prompts.append(replaceTemplateWithWord(Templates[random.randint(0, len(Templates))-1]))
print(prompts)

print("SECTION 3 COMPLETE")
print("================================================================")
###############################################################################################################################

for i in range(amountOfPrompts):
    file = open(f"{storageLocation}/videoSession{sessionNum}/video{i+1}/prompt.txt", "w")
    print(i)
    print(prompts[i])
    file.write(prompts[i])
    file.close()
    
print("SECTION 4 COMPLETE")
print("================================================================")
###############################################################################################################################

# #TODO move the min and max length to a config area
maxLength = 500
minLength = 200
removeProfanity = True
pf = ProfanityFilter()
generator = pipeline('text-generation', model=gptModel, device=0, early_stopping=False)
i = 0
while i < amountOfPrompts:
    
    file = open(f"{storageLocation}/videoSession{sessionNum}/video{i+1}/result.txt", "w")
    ttsfile = open(f"{storageLocation}/videoSession{sessionNum}/video{i+1}/tts.txt", "w")
    
    # Section of code that regenerates a story if it has profanity 
    # only works if it is told to in config, Ai can say some pretty unhinged stuff 😅 depending on the model
    # it doesnt help that most models are trained on nsfw work that would never be allowed on youtube
    def generateStory(prompt, profanity):
        global generator
        out = ""
        while out == "":
            if profanity == True:
                if useOnline == True:
                    try:
                        result = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                                {"role": "system", "content": "childrens stories based on prompts"},
                                {"role": "user", "content": prompt},
                            ], 
                        max_tokens = 100
                        )
                        out = result['choices'][0]['message']['content']
                        print(f"returned: {out}")
                        while True:
                            out = out.replace("\n\n", " ")
                            if "\n\n" in out:
                                continue
                            else:
                                break
                        print(f"after Logic: {out}")
                    except:
                        generator = pipeline('text-generation', model=gptModel, device=0, early_stopping=False)
                        out = generator(prompt, do_sample=True, min_length=minLength, max_length=maxLength)[0]['generated_text']
                        del generator
                        gc.collect()
                        torch.cuda.empty_cache()
                        
                else:
                    out = generator(prompt, do_sample=True, min_length=minLength, max_length=maxLength)[0]['generated_text']
                
                while pf.is_profane(out) or ("*" in out) or (":" in out) or (out == "") or (out == " ") or (out == "\n") or (out == prompt) or (len(out) < minLength):
                    print("\n\nProfanity/Invalid Character detected:" + out + "\n\n")
                    out = generator(prompt, do_sample=True, min_length=minLength, max_length=maxLength)[0]['generated_text']
            else:
                if useOnline == True:
                    try:
                        result = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                    {"role": "system", "content": "childrens stories based on prompts"},
                                    {"role": "user", "content": prompt},
                                ], 
                            max_tokens = 120
                        )
                        out = result['choices'][0]['message']['content']
                        while True:
                            out = out.replace("\n\n", " ")
                            if "\n\n" in out:
                                continue
                            else:
                                break
                    except:
                        generator = pipeline('text-generation', model=gptModel, device=0, early_stopping=False)
                        out = generator(prompt, do_sample=True, min_length=minLength, max_length=maxLength)[0]['generated_text']
                        del generator
                        gc.collect()
                        torch.cuda.empty_cache()
                else:
                    out = generator(prompt, do_sample=True, min_length=minLength, max_length=maxLength)[0]['generated_text']
                while ("*" in out) or (":" in out) or (out == "") or (out == " ") or (out == "\n") or (out == prompt) or (len(out) < minLength):
                    print("\n\nInvalid Character detected:" + out + "\n\n")
                    out = generator(prompt, do_sample=True, min_length=minLength, max_length=maxLength)[0]['generated_text']
        return out
    
    if useOnline == True:
        try:
            titleResult = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                        {"role": "system", "content": "one, five word youtube title from text"},
                        {"role": "user", "content": prompts[i]},
                    ], 
                max_tokens = 17
            )
            titleOut = titleResult['choices'][0]['message']['content']
        except:
            titleOut = prompts[0]
        
        while True:
            titleOut = titleOut.replace("\n", "")
            if "\n" in titleOut:
                continue
            else:
                break
        while True:
            titleOut = titleOut.replace('"', "")
            if '"' in titleOut:
                continue
            else:
                break
        while True:
            titleOut = titleOut.replace('.', "")
            if '.' in titleOut:
                continue
            else:
                break
        splitTitle = titleOut.split()
        randCaps = random.randint(0, 3)
        capIndex = []
        for capChar in range(randCaps):
            capIndex.append(random.randint(0, len(splitTitle)))
            
        for Caped in range(len(capIndex)):
            if Caped in capIndex:
                splitTitle[Caped] = splitTitle[Caped].upper()

        excited = bool(random.randint(0, 1))
        combinedTitle = " ".join(splitTitle)
        print(excited)
        if excited:
            expMark = random.randint(0, 4)
            exps = ""
            for marks in range(expMark):
                exps += ("!")
            combinedTitle = combinedTitle+exps

        print(f"TITLE - {combinedTitle}")
        
        if len(combinedTitle) > 105:
            print("THE TITLE was TOO LONG")
            sys.exit()
        
        if "SORRY" in combinedTitle:
            combinedTitle = prompts[i]
        
        if "prompt" in combinedTitle:
            combinedTitle = prompts[i]
        
        if "AI language model" in combinedTitle:
            combinedTitle = prompts[i]
        
        if "AI language" in combinedTitle:
            combinedTitle = prompts[i]
        
        if "AI" in combinedTitle:
            combinedTitle = prompts[i]
        
        if "more context" in combinedTitle:
            combinedTitle = prompts[i]
        
        if "Sorry" in combinedTitle:
            combinedTitle = prompts[i]
            
        if "Please" in combinedTitle:
            combinedTitle = prompts[i]
        
        if "please" in combinedTitle:
            combinedTitle = prompts[i]
             
        if "language model" in combinedTitle:
            combinedTitle = prompts[i]
        
        if "context" in combinedTitle:
            combinedTitle = prompts[i]
        
        if "sorry" in combinedTitle:
            combinedTitle = prompts[i]
        
        titlefile = open(f"{storageLocation}/videoSession{sessionNum}/video{i+1}/title.txt", "w")
        titlefile.write(combinedTitle)
        titlefile.close()
    else:
        combinedTitle = prompts[i]
        titlefile = open(f"{storageLocation}/videoSession{sessionNum}/video{i+1}/title.txt", "w")
        titlefile.write(combinedTitle)
        titlefile.close()
    
    out = generateStory(prompts[i], removeProfanity)
    out = out[:500]
    #print(f"OUTPUT FOR VID {i+1} ##########################")
    #print(out)
    
    out = tokenize.sent_tokenize(out)
    print(f"TOKENIZED OUTPUT FOR VID {i+1} ##########################")
    print(out)

        
    output = ""
    lines = []
    for k in out[:-1]:
        output += k + "\n"
        lines.append(k.split(" "))
    section = ""
    ttslines = []
    for k in lines:
        
        for j in k:
            if j == k[-1]:
                section += j + "\n" + " "
                ttslines.append(section)
                #print(section)
                section = ""
                break
            if len((section + j + " ")) <= lineCharacterLimit:
                section += j + " "
                continue
            else:
                ttslines.append(section)
                #print(section)
                section = ""
                section = j + " "

    
    #print(ttslines)
    ttswrite = ""
    for k in ttslines:
        #print(k)
        ttswrite += k[:-1] + "\n"
        
    #print(f"writing file {i}")
    if output == "":
        continue
    try:
        file.write(output[:-1])
        ttsfile.write(ttswrite)
    except:
        continue
    
    # RUNNING a check on the file just to double check that no errors made their way into the \
    # file as this messes with the image generation later
    
        
    ttsfile.close()
    file.close()
    File = open(f"{storageLocation}/videoSession{sessionNum}/video{i+1}/result.txt", "r")
    checkFile = File.readlines()
    print (checkFile)
    File.close()
    errored = False
    for fileCheck in checkFile:
        if fileCheck == "":
            print("FINAL FILE CHECK CAUSED ERROR 1" + fileCheck)
            errored = True
        if fileCheck == "\n":
            print("FINAL FILE CHECK CAUSED ERROR 2" + fileCheck)
            errored = True
        if fileCheck == " ":
            print("FINAL FILE CHECK CAUSED ERROR 3" + fileCheck)
            errored = True
        if fileCheck == prompts[i]:
            print("FINAL FILE CHECK CAUSED ERROR 4" + fileCheck)
            errored = True
        if ">" in fileCheck:
            print("FINAL FILE CHECK CAUSED ERROR 5" + fileCheck)
            errored = True
        
    if errored == True:
        continue
    
    i += 1

del generator
gc.collect()
torch.cuda.empty_cache()

print("SECTION 5 COMPLETE")
print("================================================================")
###############################################################################################################################

model = Text2Speech.from_pretrained(ttsModel, device ="cuda")
for i in range(amountOfPrompts):
    file = open(f"{storageLocation}/videoSession{sessionNum}/video{i+1}/result.txt", "r")
    data = file.read().split("\n")
    file.close()
    for j in range(len(data)):
        speech = model(data[j])["wav"]
        print(soundfile.write(f"{storageLocation}/videoSession{sessionNum}/video{i+1}/speech/tts{j}.wav", speech.cpu().numpy(), model.fs, "PCM_16"))
del model
gc.collect()
torch.cuda.empty_cache()

print("SECTION 6 COMPLETE")
print("================================================================")
###############################################################################################################################
# TODO add soundfonts path to a config file

tokenizer = AutoTokenizer.from_pretrained(musicModel, device=device)
model = AutoModelForSeq2SeqLM.from_pretrained(musicModel)
tokenizer = tokenizer
model = model.to(device)

max_length = 700
top_p = 0.98
temperature = 1
i = 0
#for i in range(amountOfPrompts):
while i < amountOfPrompts:
    
    tune = ""
    file = open(f"{storageLocation}/videoSession{sessionNum}/video{i+1}/result.txt", "r")
    text = file.readline()
    file.close()
    input_ids = tokenizer(text, 
                        return_tensors='pt', 
                        truncation=True, 
                        max_length=max_length)['input_ids'].to(device)

    decoder_start_token_id = model.config.decoder_start_token_id
    eos_token_id = model.config.eos_token_id

    decoder_input_ids = torch.tensor([[decoder_start_token_id]])

    for t_idx in range(max_length):
        outputs = model(input_ids=input_ids, 
        decoder_input_ids=decoder_input_ids.to(device))
        probs = outputs.logits[0][-1]
        probs = torch.nn.Softmax(dim=-1)(probs).cpu().detach().numpy()
        sampled_id = temperature_sampling(probs=top_p_sampling(probs, 
                                                            top_p=top_p, 
                                                            return_probs=True),
                                        temperature=temperature, 
                                        )
        decoder_input_ids = torch.cat((decoder_input_ids, torch.tensor([[sampled_id]])), 1)
        if sampled_id!=eos_token_id:
            sampled_token = tokenizer.decode([sampled_id])
            print(sampled_token, end="")
            tune += sampled_token
        else:
            tune += '\n'
            break
        
        
        

    # convert the ABC Notation to a Music21 stream
    try:
        stream = music21.converter.parse(tune)
        print(stream.write("midi", f"{storageLocation}/videoSession{sessionNum}/video{i+1}/music.mid"))
    except:
        print("Music Conversion Error")
        print("Trying Again")
        del outputs
        gc.collect()
        torch.cuda.empty_cache()
        continue
    # write the Music21 stream to a MIDI file
    file = random.choice(os.listdir("soundfonts"))
    f = open(f"{storageLocation}/videoSession{sessionNum}/video{i+1}/soundfont.txt", "w")
    f.write(f"soundfont used: {file}")
    f.close()
    #TODO restart ai section if the fluidsynth is frozen
    written = False
    def convertMidiToAudio():
        global written
        midi = (f'{storageLocation}/videoSession{sessionNum}/video{i+1}/music.mid')
        wav = (f'{storageLocation}/videoSession{sessionNum}/video{i+1}/music.wav')
        subprocess.run(['fluidsynth', '-ni', '-g', '0.07', f"soundfonts/{file}" , midi, '-F', wav])
        written = True
        return written

    t = Thread(target=convertMidiToAudio)
    t.daemon = True
    t.start()
    snooziness = 20
    sleep(snooziness)
    
    if written == False:
        print("writing error")
        gc.collect()
        torch.cuda.empty_cache()
        continue
    else:
        i+=1
del outputs
gc.collect()
torch.cuda.empty_cache()

print("SECTION 7 COMPLETE")
print("================================================================")
###############################################################################################################################



gc.collect()
torch.cuda.empty_cache()


pipeline = StableDiffusionWalkPipeline.from_pretrained(
    imageModel,
    torch_dtype=torch.float16,
    revision="fp16",
).to("cuda")

fps = 30
fpsScale = 14

for i in range(amountOfPrompts):
    print("startloop")
    
    
    file = open(f"{storageLocation}/videoSession{sessionNum}/video{i+1}/result.txt", "r")
    data = file.read().split("\n")
    vidLength = 0
    interpolations = []
    for j in range(len(data)):
        tts = AudioFileClip(f"{storageLocation}/videoSession{sessionNum}/video{i+1}/speech/tts{j}.wav")
        tts = tts.set_duration(tts.duration+0.1)
        value = int(round(tts.duration * 30, 0)/fpsScale)
        if value == 0:
            value = 1
        interpolations.append(value)

    file.close()
    print("fileread")
    print("generating video")
    seed = []
    for k in range(len(data)+1):
        seed.append(random.randint(0, 10000))
        
    dataPoints = [data[0]] + data
    seeds = [seed[0]] + seed
    print(interpolations)
    video_path = pipeline.walk(
        prompts=dataPoints,
        seeds=seed,
        num_interpolation_steps=interpolations,
        height=512,  # use multiples of 64 if > 512. Multiples of 8 if < 512.
        width=512,   # use multiples of 64 if > 512. Multiples of 8 if < 512.
        output_dir=f"{storageLocation}/videoSession{sessionNum}/video{i+1}",        # Where images/videos will be saved
        name='images',        # Subdirectory of output_dir where images/videos will be saved
        guidance_scale=8.5,         # Higher adheres to prompt more, lower lets model take the wheel
        num_inference_steps=20,     # Number of diffusion steps per image generated. 50 is good default
        fps=fps/fpsScale
    )
    print("generated images")
    
    gc.collect()
    torch.cuda.empty_cache()
del pipeline
gc.collect()
torch.cuda.empty_cache()

print("SECTION 8 COMPLETE")
print("================================================================")
###############################################################################################################################

for i in range(amountOfPrompts):
    file = open(f"{storageLocation}/videoSession{sessionNum}/video{i+1}/tts.txt", "r")
    data = file.read().split("\n\n")[:-1]
    file.close()
    allClips = []
    for j in range(len(data)):
        my_string = str(j)
        imageJ = my_string.zfill(6)
        tts = AudioFileClip(f"{storageLocation}/videoSession{sessionNum}/video{i+1}/speech/tts{j}.wav")
        clip = VideoFileClip(f"{storageLocation}/videoSession{sessionNum}/video{i+1}/images/images_{imageJ}/images_{imageJ}.mp4").set_duration(tts.duration+0.1)

        # swap 0 with iterator
        textData = data[j].split("\n")
        #print(textData)


        clip = clip.resize((1920, 1920))
        clip.fps=fps
        w = 1080
        h = 1920

        imageSpeed = 0
        #fl = lambda gf,t : gf(t)[int(imageSpeed*t)+w:int(imageSpeed*t),:]
        #clip = clip.fl(fl, apply_to=['mask'])
        clip = scroll(clip, h=h, w=w, x_speed=imageSpeed, y_speed=0, x_start=w, y_start=0)
        

        startTime = 0
        text = []
        for textIterations in range(len(textData)):
            tc = TextClip(textData[textIterations], 
                            fontsize = 100, 
                            font="Dosis-Bold.ttf",
                            color = 'white', 
                            align="North", 
                            method="caption",
                            stroke_color = 'black',
                            stroke_width = 5,
                            size = (w-20, h-400)).set_duration(((tts.duration)/len(textData))).set_position((10, 200)).set_start(startTime)
            text.append(tc)
            startTime += tts.duration/len(textData)



        clip = clip.set_audio(tts)
        clip = CompositeVideoClip([clip] + text)
        #clip.write_videofile(f"{storageLocation}/videoSession{sessionNum}/video{i+1}/temp/clip{j}")
        allClips.append(clip)

    #print(allClips)
    finalClip = concatenate_videoclips(allClips)
    finalClipcopy = finalClip.copy()
    music = None
    audioSet = False
    def setAudio():
        global music
        global audioSet
        global finalClip
        path = f"{storageLocation}/videoSession{sessionNum}/video{i+1}/music.wav"
        clip = finalClip
        music = AudioFileClip(path).set_duration(clip.duration)
        audioSet = True
        return audioSet
        
    print("starting thread")
    t = Thread(target=setAudio)
    t.daemon = True
    t.start()
    snooziness = 5
    sleep(snooziness)

    def backupAudio1():
        global i
        global music
        global storageLocation
        global sessionNum
        global finalClip
        global audioSet
        if i == 0:
            music = AudioFileClip(f"{storageLocation}/videoSession{sessionNum}/video{(i+1)+1}/music.wav").set_duration(finalClip.duration)
        else:
            music = AudioFileClip(f"{storageLocation}/videoSession{sessionNum}/video{(i-1)+1}/music.wav").set_duration(finalClip.duration)
        
        audioSet = True

    if audioSet == False:
        print("AUDIO TIMEOUT")
        print("starting backup thread")
        t = Thread(target=backupAudio1)
        t.daemon = True
        t.start()
        snooziness = 5
        sleep(snooziness)

    if audioSet == False:
        print("USING THE KNOWN GOOD TRACK")
        music = AudioFileClip(knownGoodMusic).set_duration(finalClip.duration)


    finalClipAudio = CompositeAudioClip([music, finalClip.audio])
    finalClip.audio = finalClipAudio
    # tc = TextClip("Everything in this video is fully automated by AI (images, music, script, tts)", 
    #             fontsize = 20, 
    #             font="Dosis-Bold.ttf",
    #             color = 'white', 
    #             align="North", 
    #             method="caption",
    #             size = (w-20, h)).set_duration(finalClip.duration).set_position((10, 100)).set_start(startTime)
    # finalClip = CompositeVideoClip([finalClip, tc])
    rendered = True
    speed = 1
    while rendered:
        try:
            finalClip.write_videofile(f"{storageLocation}/videoSession{sessionNum}/video{i+1}/finalVideo.mp4")
            rendered = False
        except:
            speed -= 0.05
            music = music.fx(vfx.speedx, speed).set_duration(finalClip.duration)
            finalClipAudio = CompositeAudioClip([music, finalClipcopy.audio])
            finalClip.audio = finalClipAudio
            continue


print("SECTION 9 COMPLETE")
print("================================================================")
###############################################################################################################################


for i in range(amountOfPrompts):
    
    file1 = open(f"{storageLocation}/videoSession{sessionNum}/video{i+1}/title.txt", "r")
    title = file1.read()
    file1.close()

    data = {
        "title": title,
        "schedule": scheduleTimes[i]
    }
    
    print(data)
    with open(f"{storageLocation}/videoSession{sessionNum}/video{i+1}/metadata.json", 'w') as f:
        json.dump(data, f)

print("SECTION 10 COMPLETE")
print("================================================================")
###############################################################################################################################


for i in range(amountOfPrompts):
    video_path = f'{storageLocation}/videoSession{sessionNum}/video{i+1}/finalVideo.mp4'
    metadata_path = f'{storageLocation}/videoSession{sessionNum}/video{i+1}/metadata.json'

    uploader = YouTubeUploader(video_path, metadata_path)
    was_video_uploaded, video_id = uploader.upload()
    print(was_video_uploaded, video_id)
print("SECTION 11 COMPLETE")
print("================================================================")
print("VIDEOS UPLOADED 😁")
###############################################################################################################################

# SECTION 12
# Get the new comments from the video using the youtube api

# SECTION 13
# Generate a reply to the comment

# SECTION 14 
# Reply to the comment with the generated comment using the youtube api
