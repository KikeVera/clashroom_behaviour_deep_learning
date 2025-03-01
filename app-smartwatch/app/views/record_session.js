import document from "document";
import { Accelerometer } from "accelerometer";
import { Gyroscope } from "gyroscope";

import * as messaging from "messaging";

export class RecordSession {
  constructor(menuObject) {
	 messaging.peerSocket.addEventListener("open", (evt) => {
		console.log("Ready to send or receive messages");
	 });
	 
	 messaging.peerSocket.addEventListener("message", (evt) => {
		if(evt.data["receptor"]==="record"){
			if(evt.data["message"]!=undefined){
				document.getElementById("action").textContent=evt.data["message"];
			}
			else{
				document.getElementById("action").textContent="Loading...";
			}
		}
	});
	 this.working=false
	 this.sensData= {accel:{x:[], y:[], z:[]}, gyro:{x:[], y:[], z:[]}} 
	 this.menuObject=menuObject;
	 this.accelerometer = new Accelerometer({ frequency: 20 });
	 this.gyroscope = new Gyroscope({ frequency: 20 });
	 
	 this.accelerometer.addEventListener("reading", () => {
		this.pushAccel();
		
	 });
	 
	 this.gyroscope.addEventListener("reading", () => {
		this.pushGyro();
	 });
	 
	 
	 
    
  }
  
  prepareView(){
	    
	
	var playButton=document.getElementById("play-button");
	playButton.addEventListener("click", ()=>{this.startSession();});
	
	var stopButton=document.getElementById("stop-button");
	stopButton.addEventListener("click", ()=>{this.stopSession();});
	
	var resetButton=document.getElementById("reset-button");
	resetButton.addEventListener("click", ()=>{this.resetSession();});
	
    
	  
  }
  
  startSession(){
	  if(this.working==false){
		  console.log("Starting session")
		  this.interval=setInterval(this.sendData.bind(this), 5000);
		  this.accelerometer.start();
		  this.gyroscope.start();
		  this.working=true
		  document.getElementById("action").textContent="Starting";		 
	  }
	  
	  
  }
  
  
  stopSession(){
	  if(this.working==true){
		  console.log("Stopping session")
		  clearInterval(this.interval);
		  this.accelerometer.stop();
		  this.gyroscope.stop();
		  this.sensData= {accel:{x:[], y:[], z:[]}, gyro:{x:[], y:[], z:[]}}
		  this.working=false
		  document.getElementById("action").textContent="Stopped";		  
	  }
	
	
	  
  }
  
  resetSession(){
	  if(this.working==false){
		  console.log("Restarting session")
		  document.getElementById("action").textContent="Reseted";
		  if (messaging.peerSocket.readyState === messaging.peerSocket.OPEN) {
				messaging.peerSocket.send({command:"reset"});
				
		  }
	  }
	  
	  
  }
  
  
  
  
  pushAccel() {
	
	 this.sensData["accel"]["x"].push(this.accelerometer.x)
	 this.sensData["accel"]["y"].push(this.accelerometer.y)
	 this.sensData["accel"]["z"].push(this.accelerometer.z)
	 
	 
	 
	 
  }
  
  pushGyro() {
	 
	 this.sensData["gyro"]["x"].push(this.gyroscope.x)
	 this.sensData["gyro"]["y"].push(this.gyroscope.y)
	 this.sensData["gyro"]["z"].push(this.gyroscope.z)
	 
	 
	 
		
   
  }
  
  sendData(){
	 
	  console.log("send data")
	  let accelx=this.sensData["accel"]["x"]
	  let accely=this.sensData["accel"]["y"]
	  let accelz=this.sensData["accel"]["z"]
	  let gyrox=this.sensData["gyro"]["x"]
	  let gyroy=this.sensData["gyro"]["y"]
	  let gyroz=this.sensData["gyro"]["z"]
	  
	  this.sensData= {accel:{x:[], y:[], z:[]}, gyro:{x:[], y:[], z:[]}}
	  
	 
	  if (messaging.peerSocket.readyState === messaging.peerSocket.OPEN) {
			messaging.peerSocket.send({sensor:accelx, command:"accelx"});
			messaging.peerSocket.send({sensor:accely, command:"accely"});
			messaging.peerSocket.send({sensor:accelz, command:"accelz"});
			messaging.peerSocket.send({sensor:gyrox, command:"gyrox"});
			messaging.peerSocket.send({sensor:gyroy, command:"gyroy"});
			messaging.peerSocket.send({sensor:gyroz, command:"gyroz"});
	  }
	  
	  
	 else {
		console.log("fallo enviando de datos")
	} 
	
	 
	    
  }
  
  
  
  
  initView() {
	  
	return document.location.assign("record_session.view");
	
	
    
  }

}

