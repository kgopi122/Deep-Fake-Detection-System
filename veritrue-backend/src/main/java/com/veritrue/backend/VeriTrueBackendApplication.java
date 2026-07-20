package com.veritrue.backend;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.ConfigurationPropertiesScan;

@SpringBootApplication
@ConfigurationPropertiesScan
public class VeriTrueBackendApplication {

    public static void main(String[] args) {
        SpringApplication.run(VeriTrueBackendApplication.class, args);
        System.out.println("✅ VeriTrue Forensic Engine is Online on Port 8000");
    }

}
