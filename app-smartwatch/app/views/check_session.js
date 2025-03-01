import document from "document";

import * as messaging from "messaging";

export class CheckSession {
  constructor(menuObject) {
	 messaging.peerSocket.addEventListener("open", (evt) => {
		console.log("Ready to send or receive messages");
	 });
	 messaging.peerSocket.addEventListener("message", (evt) => {
		if(evt.data["receptor"]==="check"){
			
			
			document.getElementById("session").textContent="Session: "+evt.data["message"]["session"];
			this.current=evt.data["message"]["number"];
			this.page=0
			if(this.session[this.page]!=undefined){
				document.getElementById("scroll").textContent=this.session[this.page];
			}
			
			else{
				document.getElementById("scroll").textContent="Loading..."
			}
			
			this.loading=false
		}
	});
	
	messaging.peerSocket.addEventListener("message", (evt) => {
		if(evt.data["receptor"]==="check_test"){
			this.session.push(evt.data["message"])
			
			
		}
	});
	
	 this.menuObject=menuObject;
	 this.deleting=false;
	 this.current=0;
	 this.page=0;
	 this.loading=false
	 this.interval = setInterval(this.restartDelButton.bind(this), 500);
	 this.session=[]
    
  }
  
  prepareView(){
	    
	
	
	var mdX
	document.getElementById('scroll').onmousedown = evt => {
		mdX=evt.screenX
		let root = document.getElementById('root')
		if(evt.screenX>root.width/2){
			this.nextPage()
		}
		else{
			this.previousPage()
		}
		
	}
	
	var playButton=document.getElementById("previous-button");
	playButton.addEventListener("click", ()=>{this.previousSession();});
	
	var stopButton=document.getElementById("next-button");
	stopButton.addEventListener("click", ()=>{this.nextSession();});
	
	var resetButton=document.getElementById("delete-button");
	resetButton.addEventListener("click", ()=>{this.deleteSession();});
	this.loadSession();


	  
  }
  
  previousSession(){
	  if(!this.loading){
		  this.current--;
		  this.loadSession();
	  }
	  
  }
  
  nextSession(){
	  this.current++;
	  this.loadSession();
	  
	  
  }
  
  
  
  deleteSession(){
	  if(!this.deleting){
		this.deleting=true
	  }
	  
	  else if(this.loading){
		  return
	  }
	  
	  else{
		this.deleting=false
		this.loading=true
		this.session=[]
		if (messaging.peerSocket.readyState === messaging.peerSocket.OPEN) {
			messaging.peerSocket.send({number:this.current, command:"delete_session"});
			document.getElementById("scroll").textContent="Deleting...";
					
		}
	  }
	  
	  
	  
  }
  
  restartDelButton(){
	  this.deleting=false
  }
  
  loadSession(){
	  this.session=[]
	  if (messaging.peerSocket.readyState === messaging.peerSocket.OPEN) {
					messaging.peerSocket.send({number:this.current, command:"previous_session"});
					
	  }
	  document.getElementById("scroll").textContent="Loading...";
		  this.loading=true
	  
  }
  
  nextPage(){
	  if(this.page<this.session.length-1){
		  this.page++;
		  document.getElementById("scroll").textContent=this.session[this.page];
	  }
	  
  }
  
  previousPage(){
	  if(this.page>0){
		  this.page--;
		  document.getElementById("scroll").textContent=this.session[this.page];
	  }
	  
	  
  }
  
  
  
  initView() {
	  
	return document.location.assign("check_session.view");
	
	
    
  }

  back() {
     return document.location.assign("index.view");
	 
  }
}

