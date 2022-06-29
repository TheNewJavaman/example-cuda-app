plugins {
    id("java")
    application
}

group = "net.javaman" // Change this to your domain, e.g. "com.google"
version = "1.0-SNAPSHOT"

// Helper to find the correct JCuda native library
fun getOsString(): String {
    val vendor = System.getProperty("java.vendor")
    return if ("The Android Project" == vendor) {
        "android"
    } else {
        var osName = System.getProperty("os.name")
        osName = osName.toLowerCase()
        when {
            osName.startsWith("windows") -> "windows"
            osName.startsWith("mac os") -> "apple"
            osName.startsWith("linux") -> "linux"
            osName.startsWith("sun") -> "sun"
            else -> "unknown"
        }
    }
}

// Helper to find the correct JCuda native library
fun getArchString(): String {
    var osArch = System.getProperty("os.arch")
    osArch = osArch.toLowerCase()
    return when {
        "i386" == osArch || "x86" == osArch || "i686" == osArch -> "x86"
        osArch.startsWith("amd64") || osArch.startsWith("x86_64") -> "x86_64"
        osArch.startsWith("arm64") -> "arm64"
        osArch.startsWith("arm") -> "arm"
        "ppc" == osArch || "powerpc" == osArch -> "ppc"
        osArch.startsWith("ppc") -> "ppc_64"
        osArch.startsWith("sparc") -> "sparc"
        osArch.startsWith("mips64") -> "mips64"
        osArch.startsWith("mips") -> "mips"
        osArch.contains("risc") -> "risc"
        else -> "unknown"
    }
}

repositories {
    mavenCentral()
}

dependencies {
    // There is/was a bug with JCuda not pulling the correct native library, so state it specifically here
    implementation("org.jcuda:jcuda:11.6.1") { isTransitive = false }
    implementation("org.jcuda:jcuda-natives:11.6.1:${getOsString() + "-" + getArchString()}")
}

application {
    mainClass.set("net.javaman.example_cuda_app.Main") // Change this to your domain, e.g. "com.google"
}