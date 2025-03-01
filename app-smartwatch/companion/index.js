import * as messaging from "messaging";


class FilesManagement {
  constructor() {
	 this.count=0
	 var dateString=new Date().toJSON()
	 this.date=dateString.slice(0,10)+"-"+dateString.slice(11,19)
	 this.sensData= {accel:{x:[], y:[], z:[]}, gyro:{x:[], y:[], z:[]}, date: this.date}
    messaging.peerSocket.addEventListener("message", (evt) => {
		this.checkCommand(evt);
	});
	
	
	
  }
	
	
	
	checkCommand(evt){
		
		if(evt.data["command"]=='previous_session'){
			var num=evt.data["number"]--;
			this.loadSession(num)
			return;
		}
		
		else if(evt.data["command"]=='next_session'){
			var num=evt.data["number"]++;
			this.loadSession(num)
			return;
		}
		
		else if(evt.data["command"]=='delete_session'){
			this.deleteSession(evt.data["number"])
			return;
		}
		
		else if(evt.data["command"]=='reset'){
			var dateString=new Date().toJSON()
			this.date=dateString.slice(0,10)+"-"+dateString.slice(11,19)
			this.sensData= {accel:{x:[], y:[], z:[]}, gyro:{x:[], y:[], z:[]}, date: this.date}
			this.count=0
			return;
		}
		
		
		else if(evt.data["command"]=='accelx'){
			this.sensData["accel"]["x"]=this.sensData["accel"]["x"].concat(evt.data["sensor"]);
			this.count++;
				
		}
		else if(evt.data["command"]=='accely'){
			this.sensData["accel"]["y"]=this.sensData["accel"]["y"].concat(evt.data["sensor"]);
			this.count++;
		}		
			
		else if(evt.data["command"]=='accelz'){
			this.sensData["accel"]["z"]=this.sensData["accel"]["z"].concat(evt.data["sensor"]);
			this.count++;
		}	
			
		else if(evt.data["command"]=='gyrox'){
			this.sensData["gyro"]["x"]=this.sensData["gyro"]["x"].concat(evt.data["sensor"]);
			this.count++;
		}		
			
		else if(evt.data["command"]=='gyroy'){
			this.sensData["gyro"]["y"]=this.sensData["gyro"]["y"].concat(evt.data["sensor"]);
			this.count++;
		}		
			
		else if(evt.data["command"]=='gyroz'){
			this.sensData["gyro"]["z"]=this.sensData["gyro"]["z"].concat(evt.data["sensor"]);
			this.count++;
		}		
				
		
		
			
		
		console.log(this.count)
		if(this.count==12){
			this.processAllData(JSON.stringify(this.sensData))
			this.count=0
			this.sensData= {accel:{x:[], y:[], z:[]}, gyro:{x:[], y:[], z:[]}, date: this.date}
			
		}
		
		
	}
	
	deleteSession(num) {
		
	
	var url="https://wldssaoih8.execute-api.eu-north-1.amazonaws.com/desarrollo_inicial/delete_session"
	console.log("deleting new session")
	console.log(num)
	
	
	var parameters = {
		method: 'POST',
		body: JSON.stringify({"number":num}),
		headers: {
			'Content-Type': 'application/json'
		}
	};
		
	fetch(url, parameters)
	.then(response => {this.loadSession(0)})
	
	
	}
	
	
	loadSession(num) {
		
	
	var url="https://wldssaoih8.execute-api.eu-north-1.amazonaws.com/desarrollo_inicial/read_session"
	console.log("loading new session")
	console.log(num)
	
	
		
	var parameters = {
		method: 'POST',
		body: JSON.stringify({"number":num}),
		headers: {
			'Content-Type': 'application/json'
		}
	};
		
	fetch(url, parameters)
	.then(response => {return response})
	.then(response => {return response["body"].getReader()})
	.then(reader => {
		reader.read().then(function processText({done, value }) {
		
		var res=JSON.parse(new TextDecoder().decode(value))
		console.log(res["number"])
		console.log(res["session"])
		console.log(res["predictions"])
		
		if(res["predictions"][0]==="Not sessions recorded"){
			messaging.peerSocket.send({receptor:"check_test", message: res["predictions"][0] });
			
		}
		else{
			var cont=0;
			var min=0;
			var pageText="Min "+(min+1)+" : "
			for (var action in(res["predictions"])){
				pageText=pageText + res["predictions"][action] + " ";
				cont++;
				if(cont==6){
					cont=0
					min++;
					messaging.peerSocket.send({receptor:"check_test", message: pageText });
					pageText="Min "+(min+1)+" : "
				}
			}
			if(pageText.length>8){
				messaging.peerSocket.send({receptor:"check_test", message: pageText });
			}
		}
		
		messaging.peerSocket.send({receptor:"check", message: {number: res["number"], session: res["session"]}});
		
		
		
			 
			
		
		})
	})
	.catch(error => {
		messaging.peerSocket.send("Error");
		console.error('Error al enviar el archivo a la función Lambda:', error);
	});
	
}
	
	
	
    processAllData(data) {
		
	
	var url="https://wldssaoih8.execute-api.eu-north-1.amazonaws.com/desarrollo_inicial/import_file"
	console.log("enviando archivos")
	console.log(data)
	
		
		
	var parameters = {
		method: 'POST',
		body: data,
		headers: {
			'Content-Type': 'application/json'
		}
	};
		
	fetch(url, parameters)
	.then(response => {return response})
	.then(response => {return response["body"].getReader()})
	.then(reader => {
		reader.read().then(function processText({done, value }) {
		
		
		console.log(JSON.parse(new TextDecoder().decode(value))["action"])
		var res=JSON.parse(new TextDecoder().decode(value))["action"]
	
		messaging.peerSocket.send({receptor:"record", message: res });
		
			 
			
		
		})
	})
	.catch(error => {
		messaging.peerSocket.send("Error");
		console.error('Error al enviar el archivo a la función Lambda:', error);
	});
	
}

  
}

new FilesManagement();
	

