import os
import json
import requests
import time
from dotenv import load_dotenv
from collections import defaultdict
from openai import OpenAI

# Hume AI API
load_dotenv()
API_KEY = os.getenv('HUME_API_KEY')
BASE_URL = 'https://api.hume.ai/v0'
BATCH_JOBS_URL = f'{BASE_URL}/batch/jobs'

# Initialize OpenAI Client
client = OpenAI()

def start_hume_job(file_path, model_type):
    with open(file_path, 'rb') as file:
        files = {'file': file}
        if model_type == 'prosody':
            data = {
                'json': json.dumps({
                    'models': {
                        model_type: {
                            'granularity': 'utterance',
                            'identify_speakers': True
                        }
                    }
                })
            }
        elif model_type == 'language':
            data = {
                'json': json.dumps({
                    'models': {
                        model_type: {
                            'granularity': 'utterance'
                        }
                    }
                })
            }
        else:
            data = {
                'json': json.dumps({
                    'models': {
                        model_type: {}
                    }
                })
            }
        headers = {'X-Hume-Api-Key': API_KEY}
        response = requests.post(BATCH_JOBS_URL, files=files, data=data, headers=headers)
    
    if response.status_code == 200:
        return response.json()['job_id']
    else:
        raise Exception(f"Failed to start job: {response.text}")

def get_job_status(job_id):
    headers = {'X-Hume-Api-Key': API_KEY}
    response = requests.get(f'{BATCH_JOBS_URL}/{job_id}', headers=headers)
    return response.json()['state']['status']

def get_job_predictions(job_id):
    headers = {'X-Hume-Api-Key': API_KEY}
    response = requests.get(f'{BATCH_JOBS_URL}/{job_id}/predictions', headers=headers)
    return response.json()

def analyze_modality(file_path, model_type):
    job_id = start_hume_job(file_path, model_type)
    print(f"Started {model_type} analysis job with ID: {job_id}")
    
    while True:
        status = get_job_status(job_id)
        print(f"{model_type.capitalize()} analysis job status: {status}")
        if status == 'COMPLETED':
            break
        time.sleep(5)
    
    predictions = get_job_predictions(job_id)
    return predictions

def write_json_to_file(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def get_top_3_emotions(emotions):
    sorted_emotions = sorted(emotions, key=lambda x: x['score'], reverse=True)
    return sorted_emotions[:3]

def synchronize_data(audio_predictions, text_predictions, video_predictions):
    synchronized_data = []
    
    text_chunks = text_predictions[0]['results']['predictions'][0]['models']['language']['grouped_predictions'][0]['predictions']
    audio_chunks = audio_predictions[0]['results']['predictions'][0]['models']['prosody']['grouped_predictions']
    video_frames = video_predictions[0]['results']['predictions'][0]['models']['face']['grouped_predictions'][0]['predictions']
    
    for audio_chunk_group in audio_chunks:
        speaker_id = audio_chunk_group['id']
        for i, audio_chunk in enumerate(audio_chunk_group['predictions']):
            chunk_data = {
                'speaker_id': speaker_id,
                'hume_transcribed_text': audio_chunk['text'],
                'youtube_transcript_text': text_chunks[i]['text'],
                'begin': audio_chunk['time']['begin'],
                'end': audio_chunk['time']['end'],
                'audio_emotions': get_top_3_emotions(audio_chunk['emotions']),
                'text_emotions': get_top_3_emotions(text_chunks[i]['emotions']),
                'video_emotions': []
            }
            
            # Find all video frames within the chunk's time range
            for frame in video_frames:
                if chunk_data['begin'] <= frame['time'] < chunk_data['end']:
                    chunk_data['video_emotions'].append({
                        'time': frame['time'],
                        'emotions': get_top_3_emotions(frame['emotions'])
                    })
            
            synchronized_data.append(chunk_data)
    
    # Sort the synchronized data by time
    synchronized_data.sort(key=lambda x: x['begin'])
    
    return synchronized_data

def write_synchronized_data_to_file(synchronized_data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for chunk in synchronized_data:
            f.write(f"Speaker ID: {chunk['speaker_id']}\n")
            f.write(f"Hume Transcribed Text: {chunk['hume_transcribed_text']}\n")
            f.write(f"Youtube Transcribed Text (what we use for the text emotions): {chunk['youtube_transcript_text']}\n")
            f.write(f"Time: {chunk['begin']:.2f}s - {chunk['end']:.2f}s\n\n")
            
            f.write("Text Top 3 Emotions:\n")
            if chunk['text_emotions']:
                for emotion in chunk['text_emotions']:
                    f.write(f"  - {emotion['name']}: {emotion['score']:.4f}\n")
            else:
                f.write("  No text emotions data available\n")
            f.write("\n")
            
            f.write("Audio Top 3 Emotions:\n")
            if chunk['audio_emotions']:
                for emotion in chunk['audio_emotions']:
                    f.write(f"  - {emotion['name']}: {emotion['score']:.4f}\n")
            else:
                f.write("  No audio emotions data available\n")
            f.write("\n")
            
            f.write("Video Emotions:\n")
            if chunk['video_emotions']:
                for frame in chunk['video_emotions']:
                    f.write(f"  Frame at {frame['time']:.2f}s:\n")
                    for emotion in frame['emotions']:
                        f.write(f"    - {emotion['name']}: {emotion['score']:.4f}\n")
            else:
                f.write("  No video emotions data available\n")
            f.write("\n" + "-"*50 + "\n\n")

def get_openai_messages(transcript, feedback_type): 
    if feedback_type == 'one-on-one':
        return [{
            "role": "system",
            "content": "You are an executive coach specializing in providing feedback to managers. Your task is to analyze conversations and provide detailed feedback. The feedback should be based on a JSON transcript that includes sentiment analysis (both verbal and facial expression). Your job is to improve the user's emotional intelligence, highlighting interactions and trends the user might have missed. The feedback should be structured to include: 1. A summary of the conversation. 2. Feedback on what was done well. 3. Feedback on what wasn't done well. 4. Insights or trends that the user may have missed."

        }, {
            "role": "user",
            "content": f"Analyze the following JSON transcript of a conversation. Provide a summary of the conversation, feedback on what was done well, and feedback on what wasn't done well. Here is the JSON transcript: {transcript}"
        }, {
          "role": "assistant",
          "content": "Ensure that the feedback is constructive and actionable, providing specific examples from the transcript when relevant. Explain what was done well, what wasn't done well, and what insights or trends the manager may have missed."
        }]
    else:
        return [{
            "role": "system",
            "content": "You are an executive coach specializing in providing feedback to anyone who does any public speaking. Your task is to analyze this presentation and provide detailed feedback. The feedback should be based on a JSON transcript that includes sentiment analysis and physical expression. Your job is to improve the user's understanding of their expression and improve emotional intelligence. You should highlight moments, trends and behaviors the user may have missed. The feedback should be structured and include what the user did well, what they should improve and any behaviors the user may have overlooked."
        }, {
            "role": "user",
            "content": f"Analyze the following JSON transcript of a presentation. Provide feedback on what was done well and what wasn't done well. Here is the JSON transcript of the video analysis: {transcript}"
        }, {
            "role": "assistant",
            "content": "Ensure that the feedback is constructive and actionable, providing specific examples from the transcript when relevant. Explain what was done well, what wasn't done well, and what behaviors the presenter may have missed."
        }]

def get_feedback(messages):
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages)

    return(completion.choices[0].message.content)

def main():
    # File paths for audio, video, and text files
    audio_path = "nvc1_audio.wav"
    video_path = "nvc1.mp4"
    transcript_path = "nvc1_transcript.txt"

    # Analyze each modality and dump API responses
    print("Analyzing audio...")
    audio_predictions = analyze_modality(audio_path, 'prosody')
    write_json_to_file(audio_predictions, "audio_predictions.json")
    
    print("Analyzing text...")
    text_predictions = analyze_modality(transcript_path, 'language')
    write_json_to_file(text_predictions, "text_predictions.json")
    
    print("Analyzing video...")
    video_predictions = analyze_modality(video_path, 'face')
    write_json_to_file(video_predictions, "video_predictions.json")
    
    print("API responses have been written to audio_predictions.json, text_predictions.json, and video_predictions.json")

    # Synchronize and process the data
    print("Synchronizing and processing data...")
    synchronized_data = synchronize_data(audio_predictions, text_predictions, video_predictions)
    
    # Write synchronized data to a text file
    output_path = "synchronized_emotions.txt"
    write_synchronized_data_to_file(synchronized_data, output_path)
    print(f"Synchronized emotion data has been written to {output_path}")

    # Send to OpenAI
    messages = get_openai_messages(synchronized_data, 'presentation')
    feedback = get_feedback(messages)
    print(feedback)

if __name__ == "__main__":
    main()
