import os
import sys
from flask import Flask, request, jsonify

# Add the current directory and testbed directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from testbed.testbed_tool import deploy_xapp_to_testbed, ensure_testbed_running
from testbed.introspection_tool import inspect_service_model_runtime

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "testbed": "running"}), 200

'''
curl -X POST http://localhost:5000/inspect \
     -H "Content-Type: application/json" \
     -d '{
       "service_model": "SLICE",
       "max_depth": 3
     }'
'''

@app.route('/inspect', methods=['POST'])
def inspect():
    data = request.json
    if not data or 'service_model' not in data:
        return jsonify({"error": "Missing service_model in request body"}), 400
    
    service_model = data['service_model']
    max_depth = data.get('max_depth', 3)
    
    print(f"Received inspection request for SM: {service_model} (depth: {max_depth})...")
    
    try:
        # Use .func to call the underlying tool logic
        result = inspect_service_model_runtime.func(service_model, max_depth=max_depth)
        return jsonify({
            "status": "SUCCESS",
            "result": result
        }), 200
    except Exception as e:
        return jsonify({
            "status": "ERROR",
            "error": str(e)
        }), 500

@app.route('/deploy', methods=['POST'])
def deploy():
    data = request.json
    if not data or 'xapp_code' not in data:
        return jsonify({"error": "Missing xapp_code in request body"}), 400
    
    xapp_code = data['xapp_code']
    artifacts = data.get('artifacts', {})
    print(f"Received deployment request for xApp code and {len(artifacts)} artifacts...")
    
    try:
        logs = deploy_xapp_to_testbed.func(xapp_code, artifacts=artifacts)
        return jsonify({
            "status": "SUCCESS",
            "logs": logs
        }), 200
    except Exception as e:
        return jsonify({
            "status": "ERROR",
            "error": str(e)
        }), 500

if __name__ == '__main__':
    # Ensure testbed is running before starting server
    print("Starting Deployer Server...")
    try:
        ensure_testbed_running()
    except Exception as e:
        print(f"Warning: Could not ensure testbed is running: {e}")
        
    app.run(host='0.0.0.0', port=5000)
