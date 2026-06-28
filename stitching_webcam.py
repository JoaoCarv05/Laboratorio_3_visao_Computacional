import cv2
import numpy as np


def stitching_webcam_realtime():
    """
    Pipeline de homografia e mosaico em tempo real com duas webcams.
    Utiliza SIFT + BFMatcher + RANSAC para alinhar e costurar os frames.

    Controles:
        q — sair
        s — salvar frame do mosaico atual
    """
    cap1 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(2)

    if not cap1.isOpened() or not cap2.isOpened():
        print("Erro: Não foi possível abrir as webcams.")
        print("Verifique se duas webcams estão conectadas.")
        if cap1.isOpened():
            print("Usando webcam 1 para ambos os feeds (modo demonstração).")
            cap2 = cap1
        else:
            return

    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()

    print("Pressione 'q' para sair.")
    print("Pressione 's' para salvar um frame.")

    frame_count = 0

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            print("Erro ao capturar frame.")
            break

        frame1 = cv2.resize(frame1, (480, 360))
        frame2 = cv2.resize(frame2, (480, 360))

        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)

        mosaico = None
        info_text = "Sem matches suficientes"

        if des1 is not None and des2 is not None and len(des1) > 2 and len(des2) > 2:
            matches_brutos = bf.knnMatch(des1, des2, k=2)

            bons_matches = []
            for m, n in matches_brutos:
                if m.distance < 0.75 * n.distance:
                    bons_matches.append(m)

            MIN_MATCH = 10

            if len(bons_matches) >= MIN_MATCH:
                src_pts = np.float32(
                    [kp1[m.queryIdx].pt for m in bons_matches]
                ).reshape(-1, 1, 2)
                dst_pts = np.float32(
                    [kp2[m.trainIdx].pt for m in bons_matches]
                ).reshape(-1, 1, 2)

                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                if H is not None:
                    inliers = int(np.sum(mask))
                    info_text = f"Matches: {len(bons_matches)} | Inliers: {inliers}"

                    img_matches = cv2.drawMatches(
                        frame1, kp1, frame2, kp2, bons_matches, None,
                        matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
                        matchesMask=mask.ravel().tolist(), flags=0
                    )

                    width = frame1.shape[1] + frame2.shape[1]
                    height = max(frame1.shape[0], frame2.shape[0])
                    warped = cv2.warpPerspective(frame1, H, (width, height))
                    mosaico = warped.copy()
                    mosaico[0:frame2.shape[0], 0:frame2.shape[1]] = frame2

                    cv2.putText(img_matches, info_text, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.imshow('Correspondencias RANSAC', img_matches)

        if mosaico is not None:
            cv2.imshow('Mosaico Tempo Real', mosaico)
        else:
            side_by_side = np.hstack([frame1, frame2])
            cv2.putText(side_by_side, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow('Mosaico Tempo Real', side_by_side)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            if mosaico is not None:
                fname = f'imagens/webcam_mosaico_{frame_count}.png'
                cv2.imwrite(fname, mosaico)
                print(f"Frame salvo: {fname}")
                frame_count += 1

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    stitching_webcam_realtime()
