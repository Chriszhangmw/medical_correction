


from flask import Flask,request,jsonify
from apps.correction_seq2seq import correction,sess
from keras.backend import  set_session
import json
import tensorflow as tf
import bjoern

app = Flask(__name__)
graph = tf.get_default_graph()


@app.route('/api/vi/medicalrecord/textCorrection',methods=['POST'])
def meke_correct():
    data = json.loads(request.get_data(as_text=True))
    if "text" not in data:
        return json.dumps({"code":401,"msg":"参数错误","data":""},ensure_ascii=False)
    text = data['text']

    global graph
    with graph.as_default():
        set_session(sess)
        result = correction.make_corrections(text)
    return jsonify({"code":100,"msg":"返回成功","data":result})


if __name__ == "__main__":
    bjoern.run(app,"0.0.0.0",5010)







