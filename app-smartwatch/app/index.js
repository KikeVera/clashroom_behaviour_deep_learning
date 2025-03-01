import document from "document";
import { RecordSession } from "./views/record_session.js";
import { CheckSession } from "./views/check_session.js";

class Menu {
  constructor() {
	this.recordSession= new RecordSession(this);
	this.checkSession= new CheckSession(this);
	this.loadButtons();
    
	
  }
  
  loadButtons(){
	  var record_session_bt = document.getElementById("record-session/start")
	 
	  var check_session_bt = document.getElementById("check-session/start")
	
	  record_session_bt.addEventListener("click", () => {this.recordSession.initView().then(this.recordSession.prepareView.bind(this.recordSession));} );
	  
	  check_session_bt.addEventListener("click", () => {this.checkSession.initView().then(this.checkSession.prepareView.bind(this.checkSession));} );
	  
  }

  
}

new Menu();
	

