import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class simplejava {
    public static void main(String[] args) {
        // Create a directory named "tmp"
        File directory = new File("tmp");
        if (!directory.exists()) {
            if (directory.mkdir()) {
                System.out.println("Directory 'tmp' created successfully.");
            } else {
                System.out.println("Failed to create directory 'tmp'.");
                return;
            }
        } else {
            System.out.println("Directory 'tmp' already exists.");
        }

        // Create a file in the "tmp" directory and write 10 digits to it
        File file = new File(directory, "digits.txt");
        try (FileWriter writer = new FileWriter(file)) {
            String digits = "0123456789";
            writer.write(digits);
            System.out.println("File 'digits.txt' created and 10 digits written successfully.");
        } catch (IOException e) {
            System.out.println("An error occurred while writing to the file.");
            e.printStackTrace();
        }
    }
}
