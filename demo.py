import requests
import json
import time
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI Client
client = OpenAI()

# Hume AI API credentials
HUME_API_KEY = os.getenv('HUME_API_KEY')

# API endpoints
BASE_URL = 'https://api.hume.ai/v0'
BATCH_JOBS_URL = f'{BASE_URL}/batch/jobs'

def start_job(file_path):
    """Start a new job for audio analysis"""
    with open(file_path, 'rb') as file:
        files = {'file': file}
        data = {
            'json':
            json.dumps({
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
        headers = {'X-Hume-Api-Key': HUME_API_KEY}
        response = requests.post(BATCH_JOBS_URL,
                                 files=files,
                                 data=data,
                                 headers=headers)

    if response.status_code == 200:
        return response.json()['job_id']
    else:
        raise Exception(f"Failed to start job: {response.text}")


def get_job_status(job_id):
    """Check the status of a job"""
    headers = {'X-Hume-Api-Key': HUME_API_KEY}
    response = requests.get(f'{BATCH_JOBS_URL}/{job_id}', headers=headers)
    return response.json()['state']['status']


def get_job_predictions(job_id):
    """Get the predictions for a completed job"""
    headers = {'X-Hume-Api-Key': HUME_API_KEY}
    response = requests.get(f'{BATCH_JOBS_URL}/{job_id}/predictions',
                            headers=headers)
    return response.json()


def extract_top_emotions(predictions, top_n=3):
    """Extract the text and top N emotions for each prediction"""
    result = []
    for prediction in predictions['predictions']:
        for grouped in prediction['models']['prosody']['grouped_predictions']:
            for pred in grouped['predictions']:
                text = pred['text']
                emotions = sorted(pred['emotions'],
                                  key=lambda x: x['score'],
                                  reverse=True)[:top_n]
                top_emotions = [(emotion['name'], round(emotion['score'], 3))
                                for emotion in emotions]
                result.append({'text': text, 'top_emotions': top_emotions})
    return result


def write_to_file(results, output_file):
    """Write the extracted results to a text file"""
    with open(output_file, 'w') as file:
        for result in results:
            file.write(f"Text: {result['text']}\n")
            for emotion, score in result['top_emotions']:
                file.write(f"  {emotion}: {score}\n")
            file.write("\n")

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
            "role": "",
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

def main(file_path, feedback_type):
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

    messages = get_openai_messages(top_emotions, feedback_type)
    feedback = get_feedback(messages)
    print(feedback)


if __name__ == "__main__":
    audio_file = input("Enter the path to your audio file: ")
    feedback_type = input("Enter the type of content (one-on-one OR presentation): ")
    main(audio_file, feedback_type)
