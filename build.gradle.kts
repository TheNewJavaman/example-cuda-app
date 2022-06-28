plugins {
    id("java")
    application
}

group = "net.javaman"
version = "1.0-SNAPSHOT"

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
    implementation("org.jcuda:jcuda:11.6.1") { isTransitive = false }
    implementation("org.jcuda:jcuda-natives:11.6.1:${getOsString() + "-" + getArchString()}")
}

application {
    mainClass.set("net.javaman.Main")
}