# Interactive Plots 

1. Running any visualization commands such as 
    ```bash
    ```
    should also place an `html` file in `/figuresWebpage/static/` such as `figuresWebpage/static/utkface_all_None_pareto_surface.html`

2. Edit `index.html`, and add figures path to the header indicated:

    ```html
    <script> 
            // Add plotly-produced file paths here. Remember to choose a unique id for each Figure. Let's use the figure number from the paper.
            $(function(){
            ("#figure1").load("static/utkface_all_None_pareto_surface.html"); 
            $("#figure2").load("static/colormnist_all_None_pareto_surface.html"); 
            });
    </script>
    ```


3. Edit body of `index.html` and reference the figures:
    ```html
    <body>
        <h1>Figures for FairPATE</h1>

        <!-- Add figures here using the ids chosen earlier -->
        <h2>Figure 1</h2>
        <div id="figure1"> </div>

        <h2>Figure 2</h2>
        <div id="figure2"> </div>
    </body>
    ```


4. See webpage with a simple webserver on port 8090 (for example):
    ```bash
    cd figuresWebpage
    python -m http.server 8090
    ```

5. Head to [http://localhost:8090](http://localhost:8090) or [http://127.0.0.1:8090](http://127.0.0.1:8090) on any browser to see the webpage.







