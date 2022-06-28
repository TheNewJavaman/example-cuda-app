# example-cuda-app

Example of JCuda (Cuda native binary called from a Java runtime application)

## Important Files

| File                                                                               | Description                     | 
|------------------------------------------------------------------------------------|---------------------------------|
| [`build.gradle.kts`](build.gradle.kts)                                             | Declares dependencies           |
| [`src/main/resources/net/javaman/main.cu`](src/main/resources/net/javaman/main.cu) | GPU code that adds two vectors  |
| [`src/main/java/net/javaman/Main.java`](src/main/java/net/javaman/Main.java)       | CPU code that runs the CPU code |