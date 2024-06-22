import requests
import json
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Hume AI API credentials
API_KEY = os.getenv('HUME_API_KEY')

# API endpoints
BASE_URL = 'https://api.hume.ai/v0'
BATCH_JOBS_URL = f'{BASE_URL}/batch/jobs'

def start_job(file_path):
    """Start a new job for audio analysis"""
    with open(file_path, 'rb') as file:
        files = {'file': file}
        data = {
            'json': json.dumps({
                'models': {
                    'prosody': {
                        'granularity': 'sentence',
                    },
                    'language': {
                        'granularity': 'word',
                    }
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
    """Check the status of a job"""
    headers = {'X-Hume-Api-Key': API_KEY}
    response = requests.get(f'{BATCH_JOBS_URL}/{job_id}', headers=headers)
    return response.json()['state']['status']

def get_job_predictions(job_id):
    """Get the predictions for a completed job"""
    headers = {'X-Hume-Api-Key': API_KEY}
    response = requests.get(f'{BATCH_JOBS_URL}/{job_id}/predictions', headers=headers)
    return response.json()

def extract_top_emotions(predictions, top_n=3):
    """Extract the text and top N emotions for each prediction"""
    result = []
    for prediction in predictions['predictions']:
        for grouped in prediction['models']['prosody']['grouped_predictions']:
            for pred in grouped['predictions']:
                text = pred['text']
                emotions = sorted(pred['emotions'], key=lambda x: x['score'], reverse=True)[:top_n]
                top_emotions = [(emotion['name'], round(emotion['score'], 3)) for emotion in emotions]
                result.append({
                    'text': text,
                    'top_emotions': top_emotions
                })
    return result

def write_to_file(results, output_file):
    """Write the extracted results to a text file"""
    with open(output_file, 'w') as file:
        for result in results:
            file.write(f"Text: {result['text']}\n")
            for emotion, score in result['top_emotions']:
                file.write(f"  {emotion}: {score}\n")
            file.write("\n")

def main(file_path):
    print("Starting emotion analysis...")
    job_id = start_job(file_path)
    print(f"Job started with ID: {job_id}")
    
    while True:
        status = get_job_status(job_id)
        print(f"Job status: {status}")
        if status == 'COMPLETED':
            break
        time.sleep(5)
    
    print("Job completed. Fetching predictions...")
    predictions = get_job_predictions(job_id)[0]["results"]
    
    top_emotions = extract_top_emotions(predictions)

    print(top_emotions)

if __name__ == "__main__":
    audio_file = input("Enter the path to your audio file: ")
    main(audio_file)