<h1>Vehicle counting 2nd Iteration</h1>
<h2>(To be updated)</h2>

<h3>Contents</h3>
<ul>
<li> Counting of Vehicles in video stream in counting_main.py.</li>
<li> Testing of counting function in counting_test.py.</li>
<li> Keys for video access not uploaded</li>
</ul>

<h3>Packages used for this process</h3>
<ul>
<li> Ultralytics package for YOLO object detection</li>
<li> Supervision package for ByteTrack object tracking and counting</li>
</ul>

<h3>Usage</h3>

<p> Open the counting_main.py file. Change the "CAM" static variable to your camera number of choice and watch the vehicles be counted both visually and logs output textually in the folder output_files. Each camera has its own output file. Each file gets re-written if counting is exited and resumed, so save files for future reference.</p>

<p> Functionality can be tested with the counting_test.py. Change the "CAM" static variable to your camera number of choice, hold ENTER to progress the video and watch the vehicles be counted both visually and logs output textually in the shared "output.txt".</p>


<p><b>NB!</b> Use requirements.txt for package dependencies in the virtual environment.</p>
<p><code>pip install -r requirements.txt  # Install dependencies from the requirements.txt file</code></p>
