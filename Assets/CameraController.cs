using System.Collections;
using System.Collections.Generic;
using UnityEngine;
//using UnityEngine.UI;

public class CameraController : MonoBehaviour {
    //=============================================================
    public WebCamTexture webCameraTexture = null;
    public GameObject plane;

    //=============================================================
    private void Init(){
		CRef();
	}

	//=============================================================
	private void CRef(){
		
	}

	//=============================================================
	private void Awake () {
		Init();
	}

	private void Start () {
        webCameraTexture = new WebCamTexture();
        plane.GetComponent<Renderer>().material.mainTexture = webCameraTexture;
        webCameraTexture.Play();
    }
	
	private void Update () {
		
	}
}