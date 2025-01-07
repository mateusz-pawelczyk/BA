# Bachelor Thesis Project



## 6. Extendibility and Future Changes

- **Adding More Libraries**: Simply clone new repos into `external/`, add a small section in `Dependencies.cmake`, and link them in the root `CMakeLists.txt`.
- **Per-Library CMake Options**: Libraries often have advanced config flags you can toggle. Keep them neatly in `Dependencies.cmake`.
- **Modular Libraries in Your Project**: If your project grows, consider structuring major components as separate libraries: