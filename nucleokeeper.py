# canvas_iterative_deconv_v2_presets.py
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
import pandas as pd
import json
from pathlib import Path

st.set_page_config(page_title="Iterative Kern-Zählung (OD + Deconv) — v2", layout="wide")
st.title("🧬 Iterative Kern-Zählung — V.2 mit Multi-Presets")

PRESET_COLORS = [
    (220, 20, 60),    # crimson
    (0, 128, 0),      # green
    (30, 144, 255),   # dodger
    (255, 165, 0),    # orange
    (148, 0, 211),    # purple
    (0, 255, 255),    # cyan
]

PRESET_FILE = "presets.json"

# -------------------- Hilfsfunktionen --------------------
def draw_scale_bar(img_disp, scale, length_orig=200, bar_height=10, margin=20, color=(0,0,0)):
    h, w = img_disp.shape[:2]
    length_disp = int(round(length_orig*scale))
    x1 = margin
    y1 = h - margin - bar_height
    x2 = x1 + length_disp
    y2 = h - margin
    cv2.rectangle(img_disp,(x1,y1),(x2,y2),color,-1)
    cv2.putText(img_disp,f"{length_orig} px",(x1,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,0.4,color,1,cv2.LINE_AA)
    return img_disp

def is_near(p1,p2,r=6):
    return np.linalg.norm(np.array(p1)-np.array(p2))<r

def dedup_new_points(candidates, existing, min_dist=6):
    out=[]
    for c in candidates:
        if not any(is_near(c,e,min_dist) for e in existing):
            out.append(c)
    return out

def extract_patch(img,x,y,radius=5):
    y_min=max(0,y-radius)
    y_max=min(img.shape[0],y+radius+1)
    x_min=max(0,x-radius)
    x_max=min(img.shape[1],x+radius+1)
    return img[y_min:y_max,x_min:x_max]

def median_od_vector_from_patch(patch, eps=1e-6):
    if patch is None or patch.size==0: return None
    patch = patch.astype(np.float32)
    OD = -np.log(np.clip((patch+eps)/255.0,1e-8,1.0))
    vec = np.median(OD.reshape(-1,3),axis=0)
    norm=np.linalg.norm(vec)
    if norm<=1e-8 or np.any(np.isnan(vec)): return None
    return (vec/norm).astype(np.float32)

def normalize_vector(v):
    v=np.array(v,dtype=float)
    n=np.linalg.norm(v)
    return (v/n).astype(float) if n>1e-12 else v

def make_stain_matrix(target_vec, hema_vec, bg_vec=None):
    t=normalize_vector(target_vec)
    h=normalize_vector(hema_vec)
    if bg_vec is None:
        bg=np.cross(t,h)
        if np.linalg.norm(bg)<1e-6:
            if abs(t[0])>0.1 or abs(t[1])>0.1:
                bg=np.array([t[1],-t[0],0.0],dtype=float)
            else:
                bg=np.array([0.0,t[2],-t[1]],dtype=float)
        bg=normalize_vector(bg)
    else:
        bg=normalize_vector(bg_vec)
    M=np.column_stack([t,h,bg]).astype(np.float32)
    M=M+np.eye(3,dtype=np.float32)*1e-8
    return M

def deconvolve(img_rgb,M):
    img=img_rgb.astype(np.float32)
    OD=-np.log(np.clip((img+1e-6)/255.0,1e-8,1.0)).reshape(-1,3)
    try:
        pinv=np.linalg.pinv(M)
        C=(pinv @ OD.T).T
    except Exception:
        return None
    return C.reshape(img_rgb.shape)

def detect_centers_from_channel_v2(channel, threshold=0.2, min_area=30, debug=False):
    arr=np.array(channel,dtype=np.float32)
    arr=np.maximum(arr,0.0)
    vmin,vmax=np.percentile(arr,[2,99.5])
    if vmax-vmin<1e-5: return [],np.zeros_like(arr,dtype=np.uint8)
    norm=(arr-vmin)/(vmax-vmin)
    norm=np.clip(norm,0.0,1.0)
    u8=(norm*255).astype(np.uint8)
    try:
        mask=cv2.adaptiveThreshold(u8,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,35,-2)
    except Exception:
        _,mask=cv2.threshold(u8,int(threshold*255),255,cv2.THRESH_BINARY)
    kernel_open=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size_open,kernel_size_open))
    kernel_close=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size_close,kernel_size_close))
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel_open)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel_close)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    centers=[]
    for cnt in contours:
        area=cv2.contourArea(cnt)
        if area>=min_area:
            M=cv2.moments(cnt)
            if M["m00"]!=0:
                cx=int(M["m10"]/M["m00"])
                cy=int(M["m01"]/M["m00"])
                centers.append((cx,cy))
    if debug:
        dbg=cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
        for (cx,cy) in centers:
            cv2.circle(dbg,(cx,cy),5,(0,0,255),-1)
        return centers,dbg
    return centers,mask

# -------------------- Session state --------------------
for k in ["groups","all_points","last_file","disp_width","C_cache","last_M_hash","history","user_presets"]:
    if k not in st.session_state:
        if k in ["groups","all_points","history"]:
            st.session_state[k]=[]
        elif k=="disp_width":
            st.session_state[k]=1000
        elif k=="user_presets":
            # lade gespeicherte Presets
            if Path(PRESET_FILE).exists():
                with open(PRESET_FILE,"r") as f:
                    st.session_state[k]=json.load(f)
            else:
                st.session_state[k] = {}
        else:
            st.session_state[k]=None

# -------------------- Upload --------------------
uploaded_file = st.file_uploader("Bild hochladen (jpg/png/tif)", type=["jpg","png","tif","tiff"])
if not uploaded_file:
    st.info("Bitte zuerst ein Bild hochladen.")
    st.stop()
if uploaded_file.name != st.session_state.last_file:
    st.session_state.groups=[]
    st.session_state.all_points=[]
    st.session_state.C_cache=None
    st.session_state.last_M_hash=None
    st.session_state.history=[]
    st.session_state.last_file=uploaded_file.name

# -------------------- Sidebar Presets --------------------
st.sidebar.markdown("### Presets (1–5)")
preset_slot = st.sidebar.selectbox("Preset Slot", [1,2,3,4,5])
preset_name = st.sidebar.text_input("Preset Name (optional)", key="preset_name")
col_preset1,col_preset2,col_preset3 = st.sidebar.columns(3)

if col_preset1.button("🔄 Lade Preset"):
    load_clicked=True
    if str(preset_slot) in st.session_state.user_presets:
        p=st.session_state.user_presets[str(preset_slot)]
        calib_radius=p["calib_radius"]
        detection_threshold=p["detection_threshold"]
        min_area_orig=p["min_area_orig"]
        dedup_dist_orig=p["dedup_dist_orig"]
        kernel_size_open=p["kernel_size_open"]
        kernel_size_close=p["kernel_size_close"]
        hema_vec0=np.array(p["hema_vec0"],dtype=float)
        aec_vec0=np.array(p["aec_vec0"],dtype=float)
        st.success(f"Preset Slot {preset_slot} geladen!")
    else:
        st.warning("Kein Preset in diesem Slot gespeichert.")

if col_preset2.button("💾 Speichern Preset"):
    st.session_state.user_presets[str(preset_slot)] = {
        "name": preset_name or f"Preset {preset_slot}",
        "calib_radius": calib_radius,
        "detection_threshold": detection_threshold,
        "min_area_orig": min_area_orig,
        "dedup_dist_orig": dedup_dist_orig,
        "kernel_size_open": kernel_size_open,
        "kernel_size_close": kernel_size_close,
        "hema_vec0": hema_vec0.tolist(),
        "aec_vec0": aec_vec0.tolist()
    }
    with open(PRESET_FILE,"w") as f:
        json.dump(st.session_state.user_presets,f,indent=2)
    st.success(f"Preset Slot {preset_slot} gespeichert!")

if col_preset3.button("🗑️ Lösche Preset"):
    if str(preset_slot) in st.session_state.user_presets:
        del st.session_state.user_presets[str(preset_slot)]
        with open(PRESET_FILE,"w") as f:
            json.dump(st.session_state.user_presets,f,indent=2)
        st.success(f"Preset Slot {preset_slot} gelöscht!")
    else:
        st.warning("Kein Preset in diesem Slot gespeichert.")

# -------------------- Sidebar Parameter (mit Defaults aus Preset) --------------------
st.sidebar.markdown("### Parameter")
calib_radius = st.sidebar.slider("Kalibrier-Radius (px, Originalbild)", 1, 30, value=st.session_state.get("calib_radius",10))
detection_threshold = st.sidebar.slider("Threshold für Detektion", 0.01,0.9,value=st.session_state.get("detection_threshold",0.2),step=0.01)
min_area_orig = st.sidebar.number_input("Minimale Konturfläche (px)",1,10000,value=st.session_state.get("min_area_orig",1000),step=1)
dedup_dist_orig = st.sidebar.number_input("Dedup-Distanz (px)",1,1000,value=st.session_state.get("dedup_dist_orig",50),step=1)
kernel_size_open = st.sidebar.slider("Kernelgröße Öffnen",1,15,value=st.session_state.get("kernel_size_open",1))
kernel_size_close = st.sidebar.slider("Kernelgröße Schließen",1,15,value=st.session_state.get("kernel_size_close",1))
circle_radius = st.sidebar.slider("Marker-Radius (px, Display)",1,12,value=5)

st.sidebar.markdown("### Startvektoren (RGB)")
hema_default=st.sidebar.text_input("Hematoxylin vector (comma)",value="0.65,0.70,0.29")
aec_default=st.sidebar.text_input("Chromogen (e.g. AEC/DAB) vector (comma)",value="0.27,0.57,0.78")
try:
    hema_vec0=np.array([float(x.strip()) for x in hema_default.split(",")],dtype=float)
    aec_vec0=np.array([float(x.strip()) for x in aec_default.split(",")],dtype=float)
except Exception:
    hema_vec0=np.array([0.65,0.70,0.29],dtype=float)
    aec_vec0=np.array([0.27,0.57,0.78],dtype=float)

# -------------------- Anzeige & Canvas --------------------
DISPLAY_WIDTH = st.slider("Anzeige-Breite (px)",300,1600,value=st.session_state.disp_width)
st.session_state.disp_width=DISPLAY_WIDTH
image_orig=np.array(Image.open(uploaded_file).convert("RGB"))
H_orig,W_orig=image_orig.shape[:2]
scale=DISPLAY_WIDTH/float(W_orig)
H_disp=int(round(H_orig*scale))
image_disp=cv2.resize(image_orig,(DISPLAY_WIDTH,H_disp),interpolation=cv2.INTER_AREA)
display_canvas=image_disp.copy()
display_canvas=draw_scale_bar(display_canvas,scale,length_orig=200,bar_height=3,color=(0,0,0))

# bestehende Punkte zeichnen
for i,g in enumerate(st.session_state.groups):
    col=tuple(int(x) for x in g.get("color",PRESET_COLORS[i%len(PRESET_COLORS)]))
    for (x_orig,y_orig) in g["points"]:
        x_disp=int(round(x_orig*scale))
        y_disp=int(round(y_orig*scale))
        cv2.circle(display_canvas,(x_disp,y_disp),circle_radius,col,-1)
    if g["points"]:
        px_disp=int(round(g["points"][0][0]*scale))
        py_disp=int(round(g["points"][0][1]*scale))
        cv2.putText(display_canvas,f"G{i+1}:{len(g['points'])}",(px_disp+6,py_disp-6),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,col,1,cv2.LINE_AA)

coords=streamlit_image_coordinates(Image.fromarray(display_canvas),
                                   key=f"clickable_image_v2_{st.session_state.last_file}",
                                   width=DISPLAY_WIDTH)

# -------------------- Sidebar Aktionen --------------------
mode=st.sidebar.radio("Aktion",["Kalibriere und zähle Gruppe (Klick)","Punkt löschen","Undo letzte Aktion"])
st.sidebar.markdown("---")
if st.sidebar.button("Reset (Alle Gruppen)"):
    st.session_state.history.append(("reset",{"groups":st.session_state.groups.copy(),"all_points":st.session_state.all_points.copy()}))
    st.session_state.groups=[]
    st.session_state.all_points=[]
    st.session_state.C_cache=None
    st.success("Zurückgesetzt.")

# -------------------- Click Handling --------------------
if coords:
    x_disp=int(coords["x"]); y_disp=int(coords["y"])
    x_orig=int(round(x_disp/scale)); y_orig=int(round(y_disp/scale))

    if mode=="Punkt löschen":
        removed=[]
        new_all=[]
        for p in st.session_state.all_points:
            if is_near(p,(x_orig,y_orig),dedup_dist_orig):
                removed.append(p)
            else:
                new_all.append(p)
        if removed:
            st.session_state.history.append(("delete_points",{"removed":removed}))
            st.session_state.all_points=new_all
            for g in st.session_state.groups:
                g["points"]=[p for p in g["points"] if not is_near(p,(x_orig,y_orig),dedup_dist_orig)]
            st.success(f"{len(removed)} Punkt(e) gelöscht.")
        else:
            st.info("Kein Punkt in der Nähe gefunden.")

    elif mode=="Undo letzte Aktion":
        if st.session_state.history:
            action,payload=st.session_state.history.pop()
            if action=="add_group":
                idx=payload["group_idx"]
                if 0<=idx<len(st.session_state.groups):
                    grp=st.session_state.groups.pop(idx)
                    for pt in grp["points"]:
                        st.session_state.all_points=[p for p in st.session_state.all_points if p!=pt]
                st.success("Letzte Gruppen-Aktion rückgängig gemacht.")
            elif action=="delete_points":
                removed=payload["removed"]
                st.session_state.all_points.extend(removed)
                st.session_state.groups.append({"vec":None,"points":removed,"color":PRESET_COLORS[len(st.session_state.groups)%len(PRESET_COLORS)]})
                st.success("Gelöschte Punkte wiederhergestellt (als neue Gruppe).")
            elif action=="reset":
                st.session_state.groups=payload["groups"]
                st.session_state.all_points=payload["all_points"]
                st.success("Reset rückgängig gemacht.")
        else:
            st.info("Keine Aktion zum Rückgängig machen.")
    else:
        # Kalibriere & zähle Gruppe
        patch=extract_patch(image_orig,x_orig,y_orig,calib_radius)
        vec=median_od_vector_from_patch(patch)
        if vec is None:
            st.warning("Patch unbrauchbar.")
        else:
            M=make_stain_matrix(vec,hema_vec0)
            M_hash=tuple(np.round(M.flatten(),6).tolist())
            recompute=st.session_state.C_cache is None or st.session_state.last_M_hash!=M_hash
            if recompute:
                C_full=deconvolve(image_orig,M)
                if C_full is None:
                    st.error("Deconvolution fehlgeschlagen.")
                    st.stop()
                st.session_state.C_cache=C_full
                st.session_state.last_M_hash=M_hash
            else:
                C_full=st.session_state.C_cache
            channel_full=C_full[:,:,0]
            centers_orig,mask=detect_centers_from_channel_v2(channel_full,
                                                              threshold=detection_threshold,
                                                              min_area=min_area_orig,
                                                              debug=False)
            new_centers=dedup_new_points(centers_orig,st.session_state.all_points,min_dist=dedup_dist_orig)
            if not any(is_near(p,(x_orig,y_orig),dedup_dist_orig) for p in st.session_state.all_points):
                new_centers.append((x_orig,y_orig))
            if new_centers:
                color=PRESET_COLORS[len(st.session_state.groups)%len(PRESET_COLORS)]
                group={"vec":vec.tolist(),"points":new_centers,"color":color}
                st.session_state.history.append(("add_group",{"group_idx":len(st.session_state.groups)}))
                st.session_state.groups.append(group)
                st.session_state.all_points.extend(new_centers)
                st.success(f"Gruppe hinzugefügt — neue Kerne: {len(new_centers)}")
            else:
                st.info("Keine neuen Kerne.")

# -------------------- Ergebnis & CSV --------------------
st.markdown("## Ergebnisse")
colA,colB=st.columns([2,1])
with colA:
    st.image(display_canvas,caption="Gezählte Kerne (Gruppenfarben)",use_column_width=True)
with colB:
    st.markdown("### Zusammenfassung")
    st.write(f"🔹 Gruppen gesamt: {len(st.session_state.groups)}")
    for i,g in enumerate(st.session_state.groups):
        st.write(f"• Gruppe {i+1}: {len(g['points'])} neue Kerne")
    st.markdown(f"**Gesamt (unique Kerne): {len(st.session_state.all_points)}**")

    if st.button("Speichere Maske (letzte Deconv-Channel)"):
        if st.session_state.C_cache is not None:
            channel_to_save=st.session_state.C_cache[:,:,0]
            vmin,vmax=np.percentile(channel_to_save,[2,99.5])
            norm=np.clip((channel_to_save-vmin)/max(1e-8,(vmax-vmin)),0.0,1.0)
            u8=(norm*255).astype(np.uint8)
            u8_disp=cv2.resize(u8,(DISPLAY_WIDTH,H_disp),interpolation=cv2.INTER_AREA)
            pil=Image.fromarray(u8_disp)
            buf=st.session_state.last_file+"_channel.png"
            pil.save(buf)
            with open(buf,"rb") as f:
                st.download_button("📥 Download Channel (PNG)",f.read(),file_name=buf,mime="image/png")
        else:
            st.info("Keine Deconvolution im Cache verfügbar.")

if st.session_state.all_points:
    rows=[]
    for i,g in enumerate(st.session_state.groups):
        for (x_orig,y_orig) in g["points"]:
            x_disp=int(round(x_orig*scale))
            y_disp=int(round(y_orig*scale))
            rows.append({"Group":i+1,"X_display":x_disp,"Y_display":y_disp,
                         "X_original":x_orig,"Y_original":y_orig})
    df=pd.DataFrame(rows)
    st.download_button("📥 CSV exportieren (Gruppen)",df.to_csv(index=False).encode("utf-8"),
                       file_name="kern_gruppen_v2.csv",mime="text/csv")
    df_unique=pd.DataFrame(st.session_state.all_points,columns=["X_original","Y_original"])
    df_unique["X_display"]=(df_unique["X_original"]*scale).round().astype(int)
    df_unique["Y_display"]=(df_unique["Y_original"]*scale).round().astype(int)
    st.download_button("📥 CSV exportieren (unique Gesamt)",df_unique.to_csv(index=False).encode("utf-8"),
                       file_name="kern_unique_v2.csv",mime="text/csv")

st.markdown("---")
st.caption("Hinweise: Deconvolution wird auf dem ORIGINALbild ausgeführt. "
           "CLAHE sollte nicht vor der Deconvolution angewendet werden. "
           "Min. Konturfläche & Dedup-Distanz werden intern auf Originalkoordinaten umgerechnet.")
