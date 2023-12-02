const imagen = document.getElementById("imagen");
const resultados = document.getElementById("resultados");
var contImg = 1;
var net;
var webcamts;
const webcam = document.getElementById("webcam");
const clasificador = knnClassifier.create();

async function reconocer(){
	webcamts = await tf.data.webcam(webcam)
}
reconocer()


async function entrenar(id){
net = await mobilenet.load();
const img = await webcamts.capture();
const activacion = net.infer(img, true);
clasificador.addExample(activacion, id);

}

async function predecir(){
	net = await mobilenet.load();
	const img = await webcamts.capture();
	const activacion = net.infer(img, "conv_preds");
	const result = await clasificador.predictClass(activacion);
	//resultados.innerHTML = JSON.stringify(result);
	const resultwo = JSON.stringify(result);
	resultados.innerHTML = resultados.innerHTML + resultwo.slice(10, 11);
	img.dispose();	
}

async function guardar(){

let modelodos = JSON.stringify(
Object.entries(clasificador.getClassifierDataset()).map(
([label, data]) => [label, Array.from(data.dataSync()), data.shape]
)
)

localStorage.setItem("modelodos", modelodos);

}

async function cargar(){

let modelodos = localStorage.getItem("modelodos");
if(modelodos == null) return;
clasificador.setClassifierDataset(Object.fromEntries(JSON.parse(modelodos).map(([label, data, shape]) => [label, tf.tensor(data, shape)])));

}

async function limpiar(){
	resultados.innerHTML = ""; 
}