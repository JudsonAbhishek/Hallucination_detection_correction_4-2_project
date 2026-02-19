from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from part3_main_pipeline import run_medhallu_pipeline
import json
import time

app = Flask(__name__, template_folder='static') # Reload trigger

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/verify", methods=["POST"])
def verify():
    mode = request.form.get("mode", "verify") 
    question = request.form.get("question")
    ai_answer = request.form.get("ai_answer")
    action = request.form.get("action", "analyze")

    print(f"\n>>> [NEW REQUEST] Mode: {mode} | Action: {action}")
    if question: print(f">>> User Question: {question}")
    if ai_answer: print(f">>> Input Text: {ai_answer[:200]}...")
    
    # --- PHASE 1: PREVIEW / REFINEMENT ---
    if action == "preview":
        from part2_llm import refine_text_for_verification, generate_refined_answer_preview
        
        if mode == "generate":
            if not question:
                return jsonify({"error": "Missing question"}), 400
            # Call Mode 1 Preview (Refine Q + Generate A)
            preview_data = generate_refined_answer_preview(question, context=ai_answer)
            return jsonify({
                "status": "PREVIEW",
                "refined_question": preview_data.get("refined_question"),
                "generated_answer": preview_data.get("generated_answer")
            })
            
        else: # mode == "verify"
            if not ai_answer:
                return jsonify({"error": "Missing text"}), 400
            # Call Mode 2 Preview (Refine Text)
            refined_text = refine_text_for_verification(ai_answer)
            return jsonify({
                "status": "PREVIEW",
                "original_text": ai_answer,
                "refined_text": refined_text
            })

    # --- PHASE 2: FINAL ANALYSIS (VERIFICATION) ---
    # The UI sends the *final* text to verify in 'ai_answer' (or 'question' + 'ai_answer' for logging)
    
    if mode == "generate":
         if not question: # Should be the refined question now
             return jsonify({"error": "Missing question"}), 400
         # In 'analyze' phase of Mode 1, 'ai_answer' contains the GENERATED answer accepted by user
         
    else: # mode == "verify"
        if not ai_answer: # Should be the refined text now
             return jsonify({"error": "Missing text"}), 400

    # Run the pipeline (common for both modes)
    try:
        result = run_medhallu_pipeline(question, ai_answer)
        result["status"] = result.get("status", "COMPLETED") 
        return jsonify(result)
    except Exception as e:
        print(f"PIPELINE CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "FAILED",
            "error": f"Internal Server Error: {str(e)}",
            "claims": []
        }), 500

if __name__ == "__main__":
    app.run(debug=True)
