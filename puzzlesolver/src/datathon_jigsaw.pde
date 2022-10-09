import java.io.*;

// Settings
String input_folder_path = "/Users/naveeniyer/Desktop/Programming/datathon_jigsaw/original_data/";
String output_folder_path = "/Users/naveeniyer/Desktop/Programming/datathon_jigsaw/output/";

int image_width = 128;
int image_height = 128;

int quadrant_width = image_width / 2;
int quadrant_height = image_height / 2;


void setup ()
{
  
  /*   UNSCRAMBLER
  File in_folder = new File(input_folder_path);
  
  int file_num = 0;
  for (File image_folder : in_folder.listFiles())
  {
    if (!image_folder.isDirectory())
    {
      continue;
    }
    
    for (File image_file : image_folder.listFiles())
    {
      if (!image_file.getName().endsWith(".png"))
      {
        continue;
      }
    
      println("Working on " + image_file.getAbsolutePath() + "...");
      PImage image = loadImage(image_file.getAbsolutePath());
      PImage quadrant1 = createImage(quadrant_width, quadrant_height, RGB);
      PImage quadrant2 = createImage(quadrant_width, quadrant_height, RGB);
      PImage quadrant3 = createImage(quadrant_width, quadrant_height, RGB);
      PImage quadrant4 = createImage(quadrant_width, quadrant_height, RGB);
      quadrant1.copy(image, 0, 0, quadrant_width, quadrant_height, 0, 0, quadrant_width, quadrant_height);
      quadrant2.copy(image, quadrant_width, 0, image_width, quadrant_height, 0, 0, quadrant_width, quadrant_height);
      quadrant3.copy(image, 0, quadrant_height, quadrant_width, image_height, 0, 0, quadrant_width, quadrant_height);
      quadrant4.copy(image, quadrant_width, quadrant_height, image_width, image_height, 0, 0, quadrant_width, quadrant_height);
      PImage[] quadrants = new PImage[4];
      
      quadrants[Integer.parseInt(image_folder.getName().substring(0, 1))] = quadrant1;
      quadrants[Integer.parseInt(image_folder.getName().substring(1, 2))] = quadrant2;
      quadrants[Integer.parseInt(image_folder.getName().substring(2, 3))] = quadrant3;
      quadrants[Integer.parseInt(image_folder.getName().substring(3, 4))] = quadrant4;
      
      
      // Create corrected image
      
      PImage correctedImage = createImage(image_width, image_height, RGB);
      correctedImage.copy(quadrants[0], 0, 0, quadrant_width, quadrant_height, 0,              0,               quadrant_width, quadrant_height);
      correctedImage.copy(quadrants[1], 0, 0, quadrant_width, quadrant_height, quadrant_width, 0              , quadrant_width, quadrant_height);
      correctedImage.copy(quadrants[2], 0, 0, quadrant_width, quadrant_height, 0,              quadrant_height, quadrant_width, quadrant_height);
      correctedImage.copy(quadrants[3], 0, 0, quadrant_width, quadrant_height, quadrant_width, quadrant_height, quadrant_width, quadrant_height);
      
      // croppedImage.save(output_folder_path + "all_correct/" + pic.getName());
      correctedImage.save(output_folder_path + "all_correct/" + file_num + ".png");
      
      file_num++;
      
      
    }
  }
  println("Done!");
  
  
  
  
  */
  
  String parent = "/Users/naveeniyer/Desktop/Programming/datathon_jigsaw/downloaded_data/128x128/";
  File[] folders = {
  new File(parent + "meme"),
  new File(parent + "landmark"),
  new File(parent + "furniture"),
  new File(parent + "illustrations")
  };
  int file_num = 0;

  int[][] permutations = {{0, 1, 2, 3}, {0, 1, 3, 2}, {0, 2, 1, 3}, {0, 2, 3, 1}, {0, 3, 1, 2}, {0, 3, 2, 1}, {1, 0, 2, 3}, {1, 0, 3, 2}, {1, 2, 0, 3}, {1, 2, 3, 0}, {1, 3, 0, 2}, {1, 3, 2, 0}, {2, 0, 1, 3}, {2, 0, 3, 1}, {2, 1, 0, 3}, {2, 1, 3, 0}, {2, 3, 0, 1}, {2, 3, 1, 0}, {3, 0, 1, 2}, {3, 0, 2, 1}, {3, 1, 0, 2}, {3, 1, 2, 0}, {3, 2, 0, 1}, {3, 2, 1, 0}};
  
  for (File image_folder : folders)
  {
    if (!image_folder.isDirectory())
    {
      continue;
    }
    
    for (File image_file : image_folder.listFiles())
    {
      if (!image_file.getName().endsWith(".png"))
      {
        continue;
      }
  
      println("Scrambling " + image_file.getName());
      PImage image = loadImage(image_file.getAbsolutePath());
      PImage quadrant1 = createImage(quadrant_width, quadrant_height, RGB);
      PImage quadrant2 = createImage(quadrant_width, quadrant_height, RGB);
      PImage quadrant3 = createImage(quadrant_width, quadrant_height, RGB);
      PImage quadrant4 = createImage(quadrant_width, quadrant_height, RGB);
      quadrant1.copy(image, 0, 0, quadrant_width, quadrant_height, 0, 0, quadrant_width, quadrant_height);
      quadrant2.copy(image, quadrant_width, 0, image_width, quadrant_height, 0, 0, quadrant_width, quadrant_height);
      quadrant3.copy(image, 0, quadrant_height, quadrant_width, image_height, 0, 0, quadrant_width, quadrant_height);
      quadrant4.copy(image, quadrant_width, quadrant_height, image_width, image_height, 0, 0, quadrant_width, quadrant_height);
      PImage[] quadrants = {quadrant1, quadrant2, quadrant3, quadrant4};
      
      
      for (int[] perm : permutations)
      {
        String perm_string = "" + perm[0] + perm[1] + perm[2] + perm[3];
        scramble(quadrants, perm).save(output_folder_path + "rescrambled/" + perm_string + "/" + file_num + ".png");
      }
      
      
      file_num++;
    }
  }
}

PImage scramble(PImage[] quadrants, int[] perm)
{
      PImage correctedImage = createImage(image_width, image_height, RGB);
      correctedImage.copy(quadrants[perm[0]], 0, 0, quadrant_width, quadrant_height, 0,              0,               quadrant_width, quadrant_height);
      correctedImage.copy(quadrants[perm[1]], 0, 0, quadrant_width, quadrant_height, quadrant_width, 0              , quadrant_width, quadrant_height);
      correctedImage.copy(quadrants[perm[2]], 0, 0, quadrant_width, quadrant_height, 0,              quadrant_height, quadrant_width, quadrant_height);
      correctedImage.copy(quadrants[perm[3]], 0, 0, quadrant_width, quadrant_height, quadrant_width, quadrant_height, quadrant_width, quadrant_height);
      
      return correctedImage;
}


/*
void draw ()
{
  File in_folder = new File(input_folder_path);
  
  for (File image_folder : in_folder.listFiles())
  {
    for (File image_file : image_folder.listFiles())
    {
      PImage image = loadImage(image_file.getAbsolutePath());
      PImage quadrant1 = createImage(quadrant_width, quadrant_height, RGB);
      PImage quadrant2 = createImage(quadrant_width, quadrant_height, RGB);
      PImage quadrant3 = createImage(quadrant_width, quadrant_height, RGB);
      PImage quadrant4 = createImage(quadrant_width, quadrant_height, RGB);
      quadrant1.copy(image, 0, 0, quadrant_width, quadrant_height, 0, 0, quadrant_width, quadrant_height);
      quadrant2.copy(image, quadrant_width, 0, image_width, quadrant_height, 0, 0, quadrant_width, quadrant_height);
      quadrant3.copy(image, 0, quadrant_height, quadrant_width, image_height, 0, 0, quadrant_width, quadrant_height);
      quadrant4.copy(image, quadrant_width, quadrant_height, image_width, image_height, 0, 0, quadrant_width, quadrant_height);
      PImage[] quadrants = {quadrant1, quadrant2, quadrant3, quadrant4};
      
      
      // Create corrected image
      
      PImage correctedImage = createImage(image_width, image_height, RGB);
      correctedImage.copy(quadrants[Integer.parseInt(image_folder.getName().substring(0, 1))], 0, 0, quadrant_width, quadrant_height, 0, 0, quadrant_width, quadrant_height);
      correctedImage.copy(quadrants[Integer.parseInt(image_folder.getName().substring(1, 2))], 0, 0, quadrant_width, quadrant_height, quadrant_width, 0, image_width, quadrant_height);
      correctedImage.copy(quadrants[Integer.parseInt(image_folder.getName().substring(2, 3))], 0, 0, quadrant_width, quadrant_height, 0, quadrant_height, quadrant_width, image_height);
      correctedImage.copy(quadrants[Integer.parseInt(image_folder.getName().substring(3, 4))], 0, 0, quadrant_width, quadrant_height, quadrant_width, quadrant_height, image_width, image_height);
      
      image(quadrant1, 0, 0);
      return;
    }
  }
}
*/
