from brisque import BRISQUE

low_brisque, high_brisque = BRISQUE("low_img.jpeg", url=False), BRISQUE("high_img.jpeg", url=False)
print(f"low_image : {low_brisque.score()}, high_image : {high_brisque.score()}")