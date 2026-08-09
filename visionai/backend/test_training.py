import urllib.request
import json
import asyncio
import websockets

async def test_training_and_ws():
    # 1. Trigger training run
    req = urllib.request.Request(
        'http://127.0.0.1:8000/api/training/start',
        data=json.dumps({
            'dataset_id': 1,
            'model_architecture': 'MobileNetV3',
            'epochs': 3,
            'batch_size': 8,
            'learning_rate': 0.001
        }).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req) as resp:
        run_data = json.loads(resp.read().decode('utf-8'))
        print('TRAINING RUN STARTED:', run_data)

    run_id = run_data['id']
    
    # 2. Connect WebSocket to monitor live metrics
    uri = f"ws://127.0.0.1:8000/api/ws/training/{run_id}"
    async with websockets.connect(uri) as websocket:
        print('CONNECTED TO WEBSOCKET FOR RUN', run_id)
        while True:
            msg = await websocket.recv()
            metrics = json.loads(msg)
            print(f"WS TELEMETRY: Status={metrics['status']}, Epoch={metrics['epoch']}/{metrics['epochs']}, TrainAcc={metrics['train_accuracy']}, ValAcc={metrics['val_accuracy']}")
            if metrics['status'] in ['COMPLETED', 'FAILED']:
                break

if __name__ == '__main__':
    asyncio.run(test_training_and_ws())
