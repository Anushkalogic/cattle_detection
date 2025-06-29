from roboflow import Roboflow

rf = Roboflow(api_key="IYnVxkCFFkQgBsrmcygz")
project = rf.workspace().project("cattle-wtx39")
model = project.version(3).model

prediction = model.predict("test.jpg").json()
print(prediction)