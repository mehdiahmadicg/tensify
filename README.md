**Tensify:**

Tensify is a real-time tension map plugin for Autodesk Maya.  
It calculates **stretch** and **compression** on your mesh as it deforms, and writes the results to vertex colors.

**Performance Optimizations in Tensify 2:**
- Accelerating Math Operations with AVX
- Memory Access with CPU Prefetching
- Implementing Adaptive Multithreading
- Reducing Expensive API Calls via Caching
- Reusing Data to Accelerate Smoothing
- Avoiding Redundant Vertex Color Updates

**Installation:**
1. If you're using **Maya 2025.3** copy **Tensify.mll** to "*your Maya installation path\bin\plug-ins*"
2. For other versions of Maya, you'll need to **compile** the plugin from the **source files** using a supported C++ compiler to generate a **.mll** file for your specific version. Once compiled, copy the file to "*your Maya installation path\bin\plug-ins*" 
3. Copy the **AEtensifyTemplate.mel** to "*your Maya installation path>\scripts\AETemplates*"
4. Load the plugin via **Maya Plugin Manager**

**How to Use:**
1. Select your mesh in Maya’s viewport
2. Run the **Tensify.py** script in the Maya Script Editor - a tensify node will be created and automatically connected to your mesh, and the stretch and compression attributes will be added to the mesh shape.

**Note on Alembic Export**:
To include the **stretch** and **compression** vertex color attributes when exporting your mesh, make sure to add both attribute names to the **Attributes** field in the **Alembic Export window**.
Refer to "img/Alembic_Export.jpg" for a visual example.

