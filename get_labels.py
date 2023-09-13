import easyocr
import regex
import cv2
import os

class GetLabelInformation:
    def __init__(self, reader):
        self.reader = reader

    def get_filtered_data(self, img, rotated):
        filtered_data = []
        x_axis = []
        y_axis = []
        title_text = []
        data = self.reader.readtext(img)
        for i in data:
            if regex.search(r"\d", i[1]):
                continue
            elif regex.search(r"[A-Za-z]", i[1]) and len(i[1])>2:
                filtered_data.append(i)
        if not rotated:
            ## for title
            title = filtered_data[0]
            t_bbox, t_text = title[0], title[1]
            t_tl, t_tr, t_br, _ = t_bbox
            mp_x_title = t_tr[0]-t_tl[0]
            mp = img.shape[1]//2
            if not mp-0.2*img.shape[1]<mp_x_title<mp+0.2*img.shape[1]:
                t_text = "not found"
            else:
                title_text.append(t_text)
            
            ## for y_axis
            sorted_filtered = sorted(filtered_data)
            y_axis_label = sorted_filtered[0]
            y_bbox, y_text = y_axis_label[0], y_axis_label[1]
            y_tl, _, y_br, _ = y_bbox
            y_axis.append(y_text)

            ## for x_axis
            x_axis_label = filtered_data[-1]
            x_bbox, x_text = x_axis_label[0], x_axis_label[1]
            x_tl, _, x_br, _ = x_bbox
            x_axis.append(x_text)
            if t_text!="not found":
                cv2.rectangle(img, t_tl, t_br, (255, 0, 0))
            cv2.rectangle(img, x_tl, x_br, (255, 0, 0))
            cv2.rectangle(img, y_tl, y_br, (255, 0, 0))
            cv2.waitKey(0)
        elif rotated:
            y_axis_label = filtered_data[0]
            # for bbox, text, _ in filtered_data[0]:
            #     (tl, tr, br, bl) = bbox
            #     tl = (int(tl[0]), int(tl[1]))
            #     tr = (int(tr[0]), int(tr[1]))
            #     br = (int(br[0]), int(br[1]))
            #     bl = (int(bl[0]), int(bl[1]))
            bbox, text = y_axis_label[0], y_axis_label[1]
            tl, _, br, _ = bbox
            y_axis.append(text)
            cv2.rectangle(img, tl, br, (255, 0, 0))
            cv2.waitKey(0)
        return x_axis, y_axis, title_text

    def run(self, img_name):
        img0 = cv2.imread(img_name)
        img90 = cv2.rotate(img0, cv2.ROTATE_90_CLOCKWISE)
        # images = [img0, img90]
        y_axis = []
        try:
            x_axis, y_axis_candidate1, title = self.get_filtered_data(img0, False)
            _, y_axis_candidate2, _ = self.get_filtered_data(img90, True)
            y_axis.append(y_axis_candidate1)
            y_axis.append(y_axis_candidate2)
            return [x_axis, y_axis, title]
        except Exception as e:
            print(e)
            return ["No textual information was detected by our OCR reader!"]

if __name__=='__main__':
    reader = easyocr.Reader(["en"])
    gl = GetLabelInformation(reader)
    img_name = './inputs/sample.png'
    print(gl.run(img_name))