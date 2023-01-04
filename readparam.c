#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  // Open the file for reading
  FILE *file = fopen("params.txt", "r");

  // Check if the file was successfully opened
  if (file == NULL) {
    perror("Error opening file");
    return 1;
  }

  // Read the values from the file
  int value1, value2, value3;
  fscanf(file, "%d %d %d", &value1, &value2, &value3);

  // Print the values that were read
  printf("Read values: %d %d %d\n", value1, value2, value3);

  // Close the file
  fclose(file);

  return 0;
}
